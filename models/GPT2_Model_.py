import pytorch_lightning as pl
from models import utils
from transformers import (
    Adafactor,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

import torch
#from Datasets import CustomDataset, Pretrain_Chunks
from Datasets_ import CustomDataset, Pretrain_Chunks
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader, ConcatDataset
from collections import Counter

import re
import string
from deepspeed.runtime.lr_schedules import WarmupDecayLR
import deepspeed
import math
import os
import csv
import torch.nn.functional as F

from models.GPT2_Model_Kadapter import GPT2LMHeadModel as GPT2_Kadapter
from models.GPT2_Model_LoRA import GPT2LMHeadModel as GPT2_Lora
from models.RecAdam import RecAdam

class GPT2(pl.LightningModule):
    def __init__(self, hparams):
        super(GPT2, self).__init__()
        self.save_hyperparameters(hparams)    

        self.unchanged_loss = 0
        self.changed_loss = 0
        self.wikipedia_loss = 0
        self.unchanged = 0
        self.changed = 0
        self.wikipedia = 0
        
        self.mix_ratio = 1
        self.mix_decay = 0.7
        self.epoch = 0

        self.model = GPT2LMHeadModel.from_pretrained(hparams.model_name_or_path)
        self.save_hyperparameters(hparams)      
        if hparams.method=='baseline' or hparams.method=='mixreview':
            self.model = GPT2LMHeadModel.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='kadapter':
            self.model = GPT2_Kadapter.from_pretrained(hparams.model_name_or_path)
            if hparams.mode != 'finetune':
                self.freeze_params(self.model) 
                for name, param in self.model.named_parameters():
                    if 'kadapter' in name or 'lm_head' in name:
                        param.requires_grad = True
        elif hparams.method=='lora':
            self.model = GPT2_Lora.from_pretrained(hparams.model_name_or_path)
            if hparams.mode != 'finetune':
                self.freeze_params(self.model) 
                for name, param in self.model.named_parameters():
                    if 'lora' in name or 'lm_head' in name:
                        param.requires_grad = True
        elif hparams.method=='recadam':
            self.model = GPT2LMHeadModel.from_pretrained(hparams.model_name_or_path)
            self.pretrained_model = GPT2LMHeadModel.from_pretrained(hparams.model_name_or_path)
            self.freeze_params(self.pretrained_model) #Freezing pretrained model
        else:
            raise Exception(f'Currently not supporting {hparams.method}')
        self.tokenizer = GPT2Tokenizer.from_pretrained(hparams.model_name_or_path)
        self.tokenizer.add_special_tokens({
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>"
            })


        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.padding_side = "left"

        self.output_dir = self.hparams.output_dir
        if self.hparams.mode=='pretrain_brute':
            self.dataset_lst = []
            lst = os.listdir(self.hparams.dataset)
            lst.sort()
            for l in lst:
                self.dataset_lst.append(self.hparams.dataset+'/'+l)
            self.dataset_index = 0
        self.global_epoch=0
        self.log('global_epoch', self.global_epoch, prog_bar=True, logger=True)

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False
        
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        
        def rid_of_specials(text):
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            return text

        return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))

    def exact_match_score(self, prediction, ground_truth):
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def _f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def calculate_scores(self, predictions, ground_truths):
        em_score = 0
        f1_score = 0
        
        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            em_score +=  self.exact_match_score(prediction, ground_truth)
            f1_score += self._f1_score(prediction, ground_truth)
        
        em_score /= len(predictions)
        f1_score /= len(predictions)
        return em_score*100, f1_score*100 

    def get_dataset(self, tokenizer, type_path, args, length=None):
        dataset = CustomDataset(tokenizer=tokenizer, type_path=type_path, input_length=args.max_input_length, 
                    output_length=args.max_output_length, args=args, length=length)
        return dataset

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False
    
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))
    

    def is_logger(self):
        return self.trainer.global_rank <= 0
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
    )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
        )

        loss = outputs[0]
        return loss

    def valid_step(self, batch):
        lm_labels = batch["label_ids"].clone().detach()
        lm_labels[source_nonprompt_mask == 0] = -100
        outputs = self(
            input_ids=batch["label_ids"],
            attention_mask=batch["label_mask"],
            lm_labels=lm_labels,
        )

        loss = outputs[0]
        print(loss)
        return loss
    
    
    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def _generative_step_finetune(self, batch, batch_idx):
        loss = self.valid_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.validation +=1
        self.validation_loss += loss
        average_loss = self.validation_loss / self.validation 
        ppl = torch.exp(average_loss)
        self.log('validation_ppl', ppl, prog_bar=True, logger=True)
        
        source = self.ids_to_clean_text(batch["source_ids"])
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            max_length=self.hparams.max_input_length + 5,
            num_beams=2,
            early_stopping=True
        )
        targets = self.ids_to_clean_text(batch["target_ids"])

        generated_ids = torch.transpose(torch.transpose(generated_ids,0,1)[self.hparams.max_input_length:],0,1)
        preds = self.ids_to_clean_text(generated_ids)
        clean_preds = []
        for text in preds:
            if "." in text:
                clean_preds.append(text[:text.find(".")+1])
            else: 
                clean_preds.append(text)
        print("clean_preds",clean_preds)
        print("targets",targets)

        em_score, f1_score = self.calculate_scores(clean_preds, targets)
        print(em_score, f1_score, ppl)
        self.log('EM score', em_score, prog_bar=True, logger=True)
        self.log('F1 score', f1_score, prog_bar=True, logger=True)
    
     
    def _generative_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if (batch_idx < (6936//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            self.unchanged +=1
            self.unchanged_loss += loss
            average_loss = self.unchanged_loss / self.unchanged 
            ppl = torch.exp(average_loss)
            self.log('Un_ppl', ppl, prog_bar=True, logger=True)
            print('Un_ppl', ppl)
        elif (batch_idx < (8713//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            self.changed +=1
            self.changed_loss += loss
            average_loss = self.changed_loss / self.changed 
            ppl = torch.exp(average_loss)
            self.log('C_ppl', ppl, prog_bar=True, logger=True)
            print('C_ppl', ppl)
        else:
            self.wikipedia +=1
            self.wikipedia_loss += loss
            average_loss = self.wikipedia_loss / self.wikipedia 
            ppl = torch.exp(average_loss)
            self.log('Kilt_Wikipedia_ppl', ppl, prog_bar=True, logger=True)
            print('Kilt_Wikipedia_ppl', ppl)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss)
        return loss

    def on_train_epoch_end(self):
        if self.hparams.mode=='pretrain_brute':
            self.dataset_index+=1
            if self.dataset_index==self.hparams.num_files:
                self.global_epoch+=1
                self.log('global_epoch', self.global_epoch, prog_bar=True, logger=True)
                self.dataset_index=0
            self.train_dataloader()
        if self.hparams.method=='mixreview':
            train_set = self.train_dataloader().dataset
        self.epoch+=1
    
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        ppl = torch.exp(loss)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if (batch_idx < (10000//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            self.log('openwebtext_ppl', ppl, prog_bar=True, logger=True)
        elif (batch_idx < (20000//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            self.log('kilt_wikipedia_ppl', ppl, prog_bar=True, logger=True)
        elif (batch_idx < (25153//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            #self.log('lambada_ppl', ppl, prog_bar=True, logger=True)
            self.predict_step(padding_length=self.hparams.max_input_length,task='lambada', batch=batch, batch_idx=batch_idx)
        elif (batch_idx < (41291//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            #self.log('lama_ppl', ppl, prog_bar=True, logger=True)
            self.predict_step(padding_length=self.hparams.max_input_length,task='lama', batch=batch, batch_idx=batch_idx)
        elif (batch_idx < (48226//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            #self.log('Un_ppl', ppl, prog_bar=True, logger=True)
            self.predict_step(padding_length=self.hparams.max_input_length,task='Unchanged', batch=batch, batch_idx=batch_idx)
        else: 
            #self.log('C_ppl', ppl, prog_bar=True, logger=True)
            self.predict_step(padding_length=self.hparams.max_input_length,task='Changed', batch=batch, batch_idx=batch_idx)

    def get_rid_of_pad(self, tokens):
        while tokens[0]==-100 or tokens[0]==50259:
            tokens.pop(0)
        return tokens

    def predict_step(self, padding_length, task, batch, batch_idx):
        source_ids = batch["source_ids"].tolist()
        target_ids = batch["target_ids"].tolist()
        batch_size = len(source_ids)
        batch_loss = 0
        batch_acc = 0
        batch_f1 = 0
        inps = []
        cont_toks_list = []
        inplens = []
        for i in range(batch_size):
            if source_ids[i]==target_ids[i]:
                context_enc = source_ids[i][:padding_length-10]
                continuation_enc = target_ids[i][padding_length-10:]
            else:
                context_enc = source_ids[i]
                continuation_enc = self.get_rid_of_pad(target_ids[i])
                #if len(continuation_enc) > 10:
                #    continuation_enc = continuation_enc[len(continuation_enc)-10:]
            # sanity check
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= self.max_length

            #inp = torch.tensor(
            #    (context_enc + continuation_enc)[-(self.max_length+1):][:-1],
            #    dtype=torch.long
            #).to(self.device)
            inp = torch.tensor(
                (context_enc + continuation_enc)[-(padding_length):][:-1],
                dtype=torch.long
            ).to(self.device)
            inplen, = inp.shape
            cont = continuation_enc

            # since in _collate we make sure length is descending, the longest is always the first one.
            #padding_length = padding_length if padding_length is not None else inplen
            # pad length from seq to padding_length
            inp = torch.cat([
                inp,  # [seq]
                torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
            ], dim=0)
            inps.append(inp.unsqueeze(0))  # [1, padding_length]
            cont_toks_list.append(cont)
            inplens.append(inplen)

        batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
        multi_logits = F.log_softmax(self._model_call(batched_inps), dim=-1).cpu()  # [batch, padding_length, vocab]
        for logits, inp, inplen, cont_toks \
                in zip(multi_logits, inps, inplens, cont_toks_list):

            # Slice to original seq length
            contlen = len(cont_toks)
            original_logits = logits
            logits = logits[inplen-contlen:inplen].unsqueeze(0)  # [1, seq, vocab]
            # Check if per-token argmax is exactly equal to continuation
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            predicted = self.ids_to_clean_text(greedy_tokens)
            ground_truth = self.ids_to_clean_text(cont_toks)
            em = self.exact_match_score(predicted[0], ground_truth[0])
            f1 = self._f1_score(predicted[0], ground_truth[0])

            # Obtain log-probs at the corresponding continuation token indices
            # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
            logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
            # Answer: (log prob, is-exact-match)
            loss = -float(logits.sum())
            if bool(max_equal) or em==1:
                batch_acc+=1
            batch_loss += loss
            batch_f1 += f1
            
        batch_loss_avg = batch_loss / batch_size
        batch_acc_avg = batch_acc / batch_size
        batch_f1_avg = batch_f1 / batch_size
        self.log(f'{task}_loss', batch_loss_avg, prog_bar=True, logger=True)
        self.log(f'{task}_acc', batch_acc_avg, prog_bar=True, logger=True)
        self.log(f'{task}_f1', batch_f1_avg, prog_bar=True, logger=True)
        return

    def configure_optimizers(self, train_len=None):
        "Prepare optimizer and schedule (linear warmup and decay)"
        if self.hparams.method=='recadam':
            no_decay = ["bias", "LayerNorm.weight"]
            model_type = 'gpt2'
            recadam_anneal_w = 1.0
            recadam_anneal_fun = 'sigmoid'
            recadam_anneal_k = 0.5
            recadam_anneal_t0 = 250
            recadam_pretrain_cof = 5000.0
            new_model = self.model
            pretrained_model = self.pretrained_model
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            not any(nd in n for nd in no_decay) and model_type in n],
                    "weight_decay": self.hparams.weight_decay,
                    "anneal_w": recadam_anneal_w,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        not any(nd in p_n for nd in no_decay) and model_type in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            not any(nd in n for nd in no_decay) and model_type not in n],
                    "weight_decay": self.hparams.weight_decay,
                    "anneal_w": 0.0,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        not any(nd in p_n for nd in no_decay) and model_type not in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            any(nd in n for nd in no_decay) and model_type in n],
                    "weight_decay": 0.0,
                    "anneal_w": recadam_anneal_w,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        any(nd in p_n for nd in no_decay) and model_type in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            any(nd in n for nd in no_decay) and model_type not in n],
                    "weight_decay": 0.0,
                    "anneal_w": 0.0,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        any(nd in p_n for nd in no_decay) and model_type not in p_n]
                }
            ]
            optimizer = RecAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,
                                anneal_fun=recadam_anneal_fun, anneal_k=recadam_anneal_k,
                                anneal_t0=recadam_anneal_t0, pretrain_cof=recadam_pretrain_cof)
        else:
            model = self.model
            
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            if self.hparams.accelerator is not None:
                optimizer = deepspeed.ops.adam.FusedAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
            else: 
                optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False)

        if self.hparams.use_lr_scheduling:
            if self.hparams.len_data==None:
                len_data = len(self.train_dataloader())
            else:
                len_data = int(self.hparams.len_data // self.hparams.train_batch_size)
            denomniator = (self.hparams.n_gpu * self.hparams.gradient_accumulation_steps)

            steps_per_epoch = ( len_data // denomniator ) + 1
            schedule_scale_factor = 1
            total_num_steps = ( steps_per_epoch * self.hparams.num_train_epochs ) * self.hparams.num_files * schedule_scale_factor

            print(f'total number of steps : {total_num_steps}')
            scheduler = WarmupDecayLR(optimizer, total_num_steps = total_num_steps ,warmup_max_lr = self.hparams.learning_rate, warmup_num_steps = int(total_num_steps * 0.1))
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "learning rate"}]
        else:
            return [optimizer]

    def train_dataloader(self): 
        if self.hparams.mode=='pretrain_brute':
            train_dataset = Pretrain_Chunks(dataset_name=self.dataset_lst[self.dataset_index],tokenizer=self.tokenizer, input_length=self.hparams.max_input_length, output_length=self.hparams.max_output_length, args=self.hparams)
        else:
            train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        if self.hparams.method=='mixreview':
            #mix_len = int(len(train_dataset) * self.mix_ratio * (self.mix_decay ** self.epoch))
            mix_len = int(len(train_dataset))
            pretrain_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="pretrain", args=self.hparams, length=mix_len)
            mixed_dataset = ConcatDataset([train_dataset,pretrain_dataset])
            print("mix len is ", mix_len)
            sampler = RandomSampler(mixed_dataset)
            dataloader = DataLoader(mixed_dataset, sampler = sampler, batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
            print("dataset length is ", len(dataloader.dataset))
        else:
            sampler = RandomSampler(train_dataset)
            dataloader = DataLoader(train_dataset, sampler=sampler,  batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        validation_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="validation", args=self.hparams,)
        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)
    
    def test_dataloader(self):
        test_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="test", args=self.hparams)
        
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            res = self.model(inps)
            return res[0][:, :, :50257]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False
        )

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [self.eot_token_id]
            else:
                context_enc = self.tok_encode(context)

            continuation_enc = self.tok_encode(continuation)
            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        for string, in tqdm(requests):
            rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                token_list=self.tok_encode(string),
                prefix_token=self.eot_token_id,
                max_seq_len=self.max_length,
                context_len=1,
            )))

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for
            # that
            string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)


        # TODO: automatic (variable) batch size detection for vectorization
        reord = utils.Reorderer(requests, _collate)
        for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length+1):][:-1],
                    dtype=torch.long
                ).to(self.device)
                inplen, = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = padding_length if padding_length is not None else inplen

                # pad length from seq to padding_length
                inp = torch.cat([
                    inp,  # [seq]
                    torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                ], dim=0)

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
            multi_logits = F.log_softmax(self._model_call(batched_inps), dim=-1).cpu()  # [batch, padding_length, vocab]
            # Make prediction directory if not exist
            pred_dir = self.pred_log.split('/')[0]
            isExist = os.path.exists(pred_dir)
            if not isExist:
                os.makedirs(pred_dir)
            #Write prediction
            with open(self.pred_log, 'a', newline='') as writefile:  
                writer = csv.writer(writefile)
                for (cache_key, _, _), logits, inp, inplen, cont_toks \
                        in zip(chunk, multi_logits, inps, inplens, cont_toks_list):

                    # Slice to original seq length
                    contlen = len(cont_toks)
                    original_logits = logits
                    logits = logits[inplen-contlen:inplen].unsqueeze(0)  # [1, seq, vocab]

                    # Check if per-token argmax is exactly equal to continuation
                    greedy_tokens = logits.argmax(dim=-1)
                    cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()
                    lines = "".join(self.ids_to_clean_text_(inp))
                    predicted = self.ids_to_clean_text_(greedy_tokens)
                    ground_truth = self.ids_to_clean_text_(cont_toks)
                    if max_equal:
                        writer.writerow([lines, ground_truth, predicted, "CORRECT"])
                    else:
                        writer.writerow([lines, ground_truth, predicted, "WRONG"])
                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))
                    # partial caching
                    """
                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                    """
                    res.append(answer)

        return reord.get_original(res)

    def greedy_until(self, requests):
        # TODO: implement fully general `until` that handles untils that are 
        #       multiple tokens or that span multiple tokens correctly

        # TODO: extract to TokenizedLM?
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        reord = utils.Reorderer(requests, _collate)

        for context, until in tqdm.tqdm(reord.get_reordered()):
            if isinstance(until, str):
                until = [until]

            primary_until, = self.tok_encode(until[0])

            context_enc = torch.tensor([self.tok_encode(context)[self.max_gen_toks - self.max_length:]]).to(self.device)

            cont = self._model_generate(context_enc, context_enc.shape[1] + self.max_gen_toks, primary_until)

            s = self.tok_decode(cont[0].tolist()[context_enc.shape[1]:])

            for term in until:
                s = s.split(term)[0]
            """
            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)
            """
            res.append(s)

        return reord.get_original(res)
    
    @utils.positional_deprecated
    def fewshot_context(self, doc, num_fewshot, provide_description=None, rnd=None, description=None):
        """ Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert rnd is not None, "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print("WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict")

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs() if self.has_validation_docs() else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = "\n\n".join(
                [self.doc_to_text(doc) + self.doc_to_target(doc) for doc in fewshotex]
            ) + "\n\n"

        example = self.doc_to_text(doc)
        return description + labeled_examples + example
    
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return jsonlines.open('data/triviaqa/unfiltered-web-train.jsonl')

    def validation_docs(self):
        return jsonlines.open('data/triviaqa/unfiltered-web-dev.jsonl')

    def fewshot_context(self, doc, num_fewshot, provide_description=None, rnd=None, description=None):
        """ Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert rnd is not None, "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print("WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict")

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs() if self.has_validation_docs() else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = "\n\n".join(
                [self.doc_to_text(doc) + self.doc_to_target(doc) for doc in fewshotex]
            ) + "\n\n"

        example = self.doc_to_text(doc)
        return description + labeled_examples + example