from torch.utils.data import Dataset, IterableDataset
import pandas as pd
import json
import random
import os

class CustomDataset(Dataset):
    def __init__(self, tokenizer, type_path, input_length, output_length, args, length=None):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path
        self.dataset_version = self.args.dataset_version
        dataset_v = ['small', 'full']   
            
        if not self.dataset_version in dataset_v:
            raise Exception(f'Provided the correct dataset version among {dataset_v}')

        # dataset for continual training
        if self.type_path=='train':
            if self.dataset_version=='small':
                self.dataset = pd.read_csv('data/recent_news_small.csv')
            elif self.args.mode == 'finetune':
                if self.args.dataset == 'data/wikipedia_09_gpt2' or self.args.dataset == 'data/wikipedia_09': 
                    total_line = 5000
                    skip = sorted(random.sample(range(1,total_line+1),total_line-256))
                    self.dataset = pd.read_csv('data/evaluation/lighttuning/08010901_lighttuning_5000.csv')
                elif self.args.dataset == 'data/wikipedia_10_gpt2' or self.args.dataset == 'data/wikipedia_10': 
                    self.dataset = pd.read_csv('data/evaluation/lighttuning/09011001_lighttuning_5000.csv')
                if self.args.dataset == 'data/wikipedia_11_gpt2' or self.args.dataset == 'data/wikipedia_11': 
                    self.dataset = pd.read_csv('data/evaluation/lighttuning/10011101_lighttuning_5000.csv')
            else:
                if self.args.dataset=='wikipedia_08':
                    self.dataset = pd.read_csv('data/wikipedia_08.csv')
                elif self.args.dataset=='wikipedia_0809':
                    self.dataset = pd.read_csv('data/wikipedia_0809_subset.csv')
                elif self.args.dataset=='wikipedia_0809_gpt2':
                    self.dataset = pd.read_csv('data/wikipedia_0809_gpt2.csv')
                elif self.args.dataset=='wikipedia_0910':
                    self.dataset = pd.read_csv('data/wikipedia_0910_subset.csv')
                elif self.args.dataset=='wikipedia_0910_gpt2':
                    self.dataset = pd.read_csv('data/wikipedia_0910_gpt2.csv')
                elif self.args.dataset=='wikipedia_1011':
                    self.dataset = pd.read_csv('data/wikipedia_1011_subset.csv')
                elif self.args.dataset=='wikipedia_1011_gpt2':
                    self.dataset = pd.read_csv('data/wikipedia_1011_gpt2.csv')
                elif self.args.dataset=='recent_news':
                    self.dataset = pd.read_csv('data/recent_news_small.csv')
                else:
                    raise Exception('The given dataset does not exist in data directory.')
        else:
            # evaluation dataset
            if self.args.check_validation_only:
                if self.args.mode == 'evaluate_ppl':
                    self.dataset = pd.read_csv('data/perplexity/'+self.args.dataset+'.csv')
                else: 
                    self.dataset = pd.read_csv('data/evaluation/aligned/'+ self.args.dataset + '.csv')
            # validation dataset
            elif self.args.dataset=='IL':
                self.dataset = pd.read_csv('data/evaluation/IL.csv')
            elif self.args.dataset=='IL_template':
                self.dataset = pd.read_csv('data/IL_template.csv')
            elif self.args.dataset=='IL_notemplate':
                self.dataset = pd.read_csv('data/IL_notemplate.csv')
            elif self.args.dataset=='data/wikipedia_09' or self.args.dataset=='wikipedia_0809' or self.args.dataset=='data/wikipedia_09_gpt2' or self.args.dataset=='wikipedia_0809_gpt2':
                df1 = pd.read_csv('data/evaluation/aligned/0801-0901_unchanged.csv')
                df2 = pd.read_csv('data/evaluation/aligned/0801-0901_updated.csv')
                df3 = pd.read_csv('data/evaluation/aligned/0801-0901_new.csv')
                df4 = pd.read_csv('data/evaluation/IL.csv')
                df1 = pd.concat([df1, df2])
                df1 = pd.concat([df1, df3])
                self.dataset = df1
                # self.dataset = pd.concat([df1, df4])
            elif self.args.dataset=='data/wikipedia_10_gpt2' or self.args.dataset=='wikipedia_0910_gpt2':
                df1 = pd.read_csv('data/evaluation/aligned/0901-1001_unchanged.csv')
                df2 = pd.read_csv('data/evaluation/aligned/0901-1001_updated.csv')
                df3 = pd.read_csv('data/evaluation/aligned/0901-1001_new.csv')
                df4 = pd.read_csv('data/evaluation/IL.csv')
                df1 = pd.concat([df1, df2])
                df1 = pd.concat([df1, df3])
                self.dataset = df1
                # self.dataset = pd.concat([df1, df4])
            elif self.args.dataset=='UnC_09':
                self.dataset = pd.read_csv('data/UnL_0809.csv')
            elif self.args.dataset=='NL_09':
                self.dataset = pd.read_csv('data/NL_0809.csv')
            elif self.args.dataset=='UL_09':
                self.dataset = pd.read_csv('data/UpL_0809.csv')
            else:
                self.dataset = pd.read_csv('data/evaluation/IL.csv')
        
        print(f'Length of dataset retrieving is.. {len(self.dataset)}')
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        # continual pretraining
        input_nonprompt = None
        label_ = None
        ppl_input = None
        if self.type_path=='validation' and ('gpt2' in self.args.model_name_or_path):
            if self.args.mode == 'evaluate_ppl':
                input_ = example_batch['text']
                target_ = example_batch['text']
            else: 
                s = example_batch['subject']
                r = example_batch['relation']
                o = example_batch['objective']
                if self.args.mode == 'evaluate_ppl':
                    input_ = s + ' ' + r + ' ' + o
                    input_nonprompt = o
                    target_ = s + ' ' + r + ' ' + o 
                elif self.args.mode == 'evaluate':
                    input_ = s + ' ' + r
                    target_ = o
                else: 
                    label_ = s + ' ' + r + ' ' + o 
                    target_ = o
                    input_ = s + ' ' + r
                    input_nonprompt = o
        elif self.type_path=='validation' and ('t5' in self.args.model_name_or_path):
            if self.args.mode == 'evaluate_ppl':
                input_ = example_batch['text']
                target_ = example_batch['text']
            else: 
                s = example_batch['subject']
                r = example_batch['relation']
                input_ = r.capitalize() + ' of ' + s + ' is' + ' <extra_id_0> .' 
                # input_ = s + ' ' + r + ' <extra_id_0> .' 
                target_ = example_batch['objective']
        elif 'gpt2' in self.args.model_name_or_path:
            if self.args.mode == 'finetune':
                s = example_batch['subject']
                r = example_batch['relation']
                o = example_batch['objective']  
                input_ = s + ' ' + r + ' ' + o 
                target_ = s + ' ' + r + ' ' + o 
                label_ = s + ' ' + r + ' ' + o
            else: 
                input_ = example_batch['text']
                target_ = example_batch['text']
        elif 't5' in self.args.model_name_or_path:
            if self.args.mode == 'finetune':
                s = example_batch['subject']
                r = example_batch['relation']
                # input_ = s + ' ' + r + ' <extra_id_0> .'
                input_ = r.capitalize() + ' of ' + s + ' is' + ' <extra_id_0> .'  
                target_ = example_batch['objective']
            else: 
                input_ = example_batch['text']
                target_ = example_batch['text']
        else:
            raise Exception('Model should either T5 or GPT2.')
        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt") 
        if input_nonprompt is not None:
            input_nonprompt = self.tokenizer.batch_encode_plus([str(input_nonprompt)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt") 
        if label_ is not None:
            label_ = self.tokenizer.batch_encode_plus([str(label_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        if ppl_input is not None:
            ppl_input = self.tokenizer.batch_encode_plus([str(ppl_input)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")               
        
        return source, targets, input_nonprompt, label_, ppl_input
  
    def __getitem__(self, index):
        source, targets, input_nonprompt, label, ppl_input = self.convert_to_features(self.dataset.iloc[index]) 
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        if input_nonprompt is not None:
            source_nonprompt_mask = input_nonprompt["attention_mask"].squeeze()
        else: 
            source_nonprompt_mask = -1
        
        if label is not None:
            label_ids = label["input_ids"].squeeze()
            label_mask = label["attention_mask"].squeeze()
        else: 
            label_ids = -1
            label_mask = -1

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "source_nonprompt_mask": source_nonprompt_mask, "label_ids": label_ids, "label_mask": label_mask}
    
class Pretrain_Chunks(Dataset):
    def __init__(self, dataset_name, tokenizer, input_length, output_length, args):
        self.args = args
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.output_length = output_length
        self.dataset = pd.read_csv(dataset_name)
        print(f'Getting dataset {dataset_name} with length {len(self.dataset)}')
        
    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        # continual pretraining
        if 'gpt2' in self.args.model_name_or_path:
            input_ = example_batch['text']
            target_ = example_batch['text']
        else:
            input_ = example_batch['input']
            target_ = example_batch['output']
        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")                          
        return source, targets
    
    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset.iloc[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}