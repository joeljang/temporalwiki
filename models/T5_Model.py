import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import (
    Adafactor,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import torch
from Datasets import CustomDataset, Pretrain_Chunks
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, ConcatDataset
from rouge import Rouge
from collections import Counter

import re
import string
import copy
import os
import random
import csv

from deepspeed.runtime.lr_schedules import WarmupDecayLR
import deepspeed

from models.T5_Model_CL import T5ForConditionalGeneration as T5_Kadapter

class T5(pl.LightningModule):
    def __init__(self, hparams):
        super(T5, self).__init__()
        self.save_hyperparameters(hparams)
        
        if hparams.method=='baseline':
            self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='kadapter':
            self.model = T5_Kadapter.from_pretrained(hparams.model_name_or_path)
            self.freeze_params(self.model.get_encoder()) #Freezing the encoder
            # Unfreezing the parameters used for kadapters in encoder
            for name, param in self.model.named_parameters():
                if 'kadapter' in name:
                    param.requires_grad = True
        else:
            raise Exception('Currently not supporting {hparams.method}')
        
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
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
            text = text.replace("<extra_id_2>", "")
            text = text.replace("<extra_id_3>", "")
            return text

        return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))

    def exact_match_score(self, prediction, ground_truth):
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def accuracy_match_score(self, prediction, ground_truth):
        return int(prediction.strip() == ground_truth.strip())

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
        accuracy = 0
        
        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            em_score +=  self.exact_match_score(prediction, ground_truth)
            accuracy += self.accuracy_match_score(prediction, ground_truth)
        
        em_score /= len(predictions)
        accuracy /= len(predictions)
        return em_score*100, accuracy*100

    def calculate_f1_scores(self, predictions, ground_truths):
        f1_score = 0 
        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            f1_score += self._f1_score(prediction, ground_truth)

        f1_score /= len(predictions)
        return f1_score*100

    def get_dataset(self, tokenizer, type_path, args, length=None):
        dataset = CustomDataset(tokenizer=tokenizer, type_path=type_path, input_length=args.max_input_length, 
                        output_length=args.max_output_length, args=args, length=length)
        return dataset
             
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))
    

    def is_logger(self):
        return self.trainer.global_rank <= 0
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
    )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss
    
    
    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)
    
     
    def _generative_step(self, batch, batch_idx):     
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=10,
            num_beams=2,
            early_stopping=True
        )
        
        preds = self.ids_to_clean_text(generated_ids)
        targets = self.ids_to_clean_text(batch["target_ids"])
        source = self.ids_to_clean_text(batch["source_ids"])
        print("preds", preds)
        print("targets", targets)
        loss = self._step(batch)

        em_score = 0
        accuracy = 0
        f1_score = 0

        em_score, accuracy = self.calculate_scores(preds, targets)
        f1_score = self.calculate_f1_scores(preds, targets)

        em_score = torch.tensor(em_score,dtype=torch.float32)
        accuracy = torch.tensor(accuracy,dtype=torch.float32)
        f1_score = torch.tensor(f1_score, dtype=torch.float32)

        if (batch_idx < (10000//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            self.log('UnL_loss', loss, prog_bar=True, logger=True)
            self.log('UnL_EM', em_score, prog_bar=True, logger=True)
            self.log('UnL_F1', f1_score, prog_bar=True, logger=True)
        elif (batch_idx < (10785//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            self.log('UL_loss', loss, prog_bar=True, logger=True)
            self.log('UL_EM', em_score, prog_bar=True, logger=True)
            self.log('UL_F1', f1_score, prog_bar=True, logger=True)
        elif (batch_idx < (12329//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            self.log('NL_loss', loss, prog_bar=True, logger=True)
            self.log('NL_EM', em_score, prog_bar=True, logger=True)
            self.log('NL_F1', f1_score, prog_bar=True, logger=True)
        else:
            self.log('IL_loss', loss, prog_bar=True, logger=True)
            self.log('IL_EM', em_score, prog_bar=True, logger=True)
            self.log('IL_F1', f1_score, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss)
        return loss

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
    
    def on_train_epoch_end(self):
        if self.hparams.mode=='pretrain_brute':
            self.dataset_index+=1
            if self.dataset_index==self.hparams.num_files:
                self.global_epoch+=1
                self.log('global_epoch', self.global_epoch, prog_bar=True, logger=True)
                self.dataset_index=0
            self.train_dataloader()

    def train_dataloader(self): 
        if self.hparams.mode=='pretrain_brute':
            train_dataset = Pretrain_Chunks(dataset_name=self.dataset_lst[self.dataset_index],tokenizer=self.tokenizer, input_length=self.hparams.max_input_length, output_length=self.hparams.max_output_length, args=self.hparams)
        else:
            train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(train_dataset, sampler=sampler,  batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        #dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        validation_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="validation", args=self.hparams,)
        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)
    
    def test_dataloader(self):
        test_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="test", args=self.hparams)
        
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)