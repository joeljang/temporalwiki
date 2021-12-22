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
            else:
                if self.args.dataset=='wikipedia_08':
                    self.dataset = pd.read_csv('data/wikipedia_08.csv')
                elif self.args.dataset=='wikipedia_0809':
                    self.dataset = pd.read_csv('data/wikipedia_0809_subset.csv')
                elif self.args.dataset=='wikipedia_0910':
                    self.dataset = pd.read_csv('data/wikipedia_0910_subset.csv')
                elif self.args.dataset=='wikipedia_1011':
                    self.dataset = pd.read_csv('data/wikipedia_1011_subset.csv')
                elif self.args.dataset=='recent_news':
                    self.dataset = pd.read_csv('data/recent_news_small.csv')
                else:
                    raise Exception('The given dataset does not exist in data directory.')
        else:
            if self.args.dataset=='IL':
                self.dataset = pd.read_csv('data/IL.csv')
            elif self.args.dataset=='IL_template':
                self.dataset = pd.read_csv('data/IL_template.csv')
            elif self.args.dataset=='IL_notemplate':
                self.dataset = pd.read_csv('data/IL_notemplate.csv')
            elif self.args.dataset=='data/wikipedia_09' or self.args.dataset=='wikipedia_0809' or self.args.dataset=='data/20210901_gpt2':
                df1 = pd.read_csv('data/UnL_0809.csv')
                df2 = pd.read_csv('data/UpL_0809.csv')
                df3 = pd.read_csv('data/NL_0809.csv')
                df1 = pd.concat([df1, df2])
                self.dataset = pd.concat([df1, df3])
            elif self.args.dataset=='UnC_09':
                self.dataset = pd.read_csv('data/UnL_0809.csv')
            elif self.args.dataset=='NL_09':
                self.dataset = pd.read_csv('data/NL_0809.csv')
            elif self.args.dataset=='UL_09':
                self.dataset = pd.read_csv('data/UpL_0809.csv')
            else:
                self.dataset = pd.read_csv('data/IL.csv')
        
        print(f'Length of dataset retrieving is.. {len(self.dataset)}')
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        # continual pretraining
        if 'gpt2' in self.args.model_name_or_path:
            s = example_batch['subject']
            r = example_batch['relation']
            o = example_batch['objective']
            input_ = s + ' ' + r + ' ' + o 
            target_ = s + ' ' + r + ' ' + o 
        elif self.type_path=='validation' and (self.args.dataset=='data/wikipedia_09' or self.args.dataset=='wikipedia_0809'):
            s = example_batch['subject']
            r = example_batch['relation']
            input_ = s + ' ' + r + ' <extra_id_0> .' 
            target_ = example_batch['objective']
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