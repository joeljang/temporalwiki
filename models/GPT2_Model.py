import pytorch_lightning as pl

from transformers import (
    Adafactor,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

import torch
from Datasets import CustomDataset, Pretrain_Chunks
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

from models.GPT2_Model_CL import GPT2LMHeadModel as GPT2_Kadapter

class GPT2(pl.LightningModule):
    def __init__(self, hparams):
        super(GPT2, self).__init__()
        self.save_hyperparameters(hparams)    
        self.unchanged_loss = 0
        self.updated_loss = 0
        self.new_loss = 0
        self.invariant_loss = 0
        self.unchanged = 0
        self.updated = 0
        self.new = 0
        self.invariant = 0

        self.model = GPT2LMHeadModel.from_pretrained(hparams.model_name_or_path)
        self.save_hyperparameters(hparams)      
        if hparams.method=='baseline':
            self.model = GPT2LMHeadModel.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='kadapter':
            self.model = GPT2_Kadapter.from_pretrained(hparams.model_name_or_path)
            self.freeze_params(self.model) 
            for name, param in self.model.named_parameters():
                if 'kadapter' in name or 'lm_head' in name:
                    param.requires_grad = True
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
        if type_path=='validation':
            dataset = CustomDataset(tokenizer=tokenizer, type_path=type_path, input_length=args.max_input_length, 
                        output_length=args.max_output_length, args=args, length=length)
        else:
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
        source_nonprompt_mask = batch['source_nonprompt_mask']
        # print(source_nonprompt_mask)
        lm_labels[source_nonprompt_mask == 0] = -100
        # lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        # print(lm_labels, batch["label_ids"])
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
    
     
    def _generative_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # source = self.ids_to_clean_text(batch["source_ids"])
        # generated_ids = self.model.generate(
        #     batch["source_ids"],
        #     attention_mask=batch["source_mask"],
        #     use_cache=True,
        #     max_length=self.hparams.max_input_length + 3,
        #     num_beams=2,
        #     early_stopping=True
        # )
        # generated_ids = torch.transpose(torch.transpose(generated_ids,0,1)[self.hparams.max_input_length:],0,1)
        # preds = self.ids_to_clean_text(generated_ids)
        # clean_preds = []
        # for text in preds:
        #     if "." in text:
        #         clean_preds.append(text[:text.find(".")+1])
        #     else: 
        #         clean_preds.append(text)
        # print("clean_preds",clean_preds)
        # targets = self.ids_to_clean_text(batch["target_ids"])
        # print("targets",targets)

        # if self.hparams.mode == 'finetune':
        #     with open(self.hparams.output_log, 'a', newline='') as writefile: 
        #         writer = csv.writer(writefile)
        #         for i in range(len(targets)):
        #             writer.writerow([source[i], clean_preds[i], targets[i], self.exact_match_score(clean_preds[i], targets[i])])
        # em_score, f1_score = self.calculate_scores(clean_preds, targets)

        if (batch_idx < (10000//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            self.unchanged +=1
            self.unchanged_loss += loss
            average_loss = self.unchanged_loss / self.unchanged 
            ppl = torch.exp(average_loss)
            self.log('UnL_ppl', ppl, prog_bar=True, logger=True)
            print('UnL_ppl', ppl)
            # self.log('UnL_EM', em_score, prog_bar=True, logger=True)
            # self.log('UnL_F1', f1_score, prog_bar=True, logger=True)
        elif (batch_idx < (15000//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            self.updated +=1
            self.updated_loss += loss
            average_loss = self.updated_loss / self.updated 
            ppl = torch.exp(average_loss)
            self.log('UL_ppl', ppl, prog_bar=True, logger=True)
            print('UL_ppl', ppl)
            # self.log('UL_EM', em_score, prog_bar=True, logger=True)
            # self.log('UL_F1', f1_score, prog_bar=True, logger=True)
        elif (batch_idx < (20000//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            self.new +=1
            self.new_loss += loss
            average_loss = self.new_loss / self.new 
            ppl = torch.exp(average_loss)
            self.log('NL_ppl', ppl, prog_bar=True, logger=True)
            print('NL_ppl', ppl)
            # self.log('NL_EM', em_score, prog_bar=True, logger=True)
            # self.log('NL_F1', f1_score, prog_bar=True, logger=True)
        else:
            self.invariant +=1
            self.invariant_loss += loss
            average_loss = self.invariant_loss / self.invariant 
            ppl = torch.exp(average_loss)
            self.log('IL_ppl', ppl, prog_bar=True, logger=True)
            print('IL_ppl', ppl)
            # self.log('IL_EM', em_score, prog_bar=True, logger=True)
            # self.log('IL_F1', f1_score, prog_bar=True, logger=True)
        
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

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch, batch_idx)

    def configure_optimizers(self, train_len=None):
        "Prepare optimizer and schedule (linear warmup and decay)"
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
            schedule_scale_factor = 6
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
        sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(train_dataset, sampler=sampler,  batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        validation_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="validation", args=self.hparams,)
        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)
    
    def test_dataloader(self):
        test_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="test", args=self.hparams)
        
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)