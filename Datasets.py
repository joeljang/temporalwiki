from torch.utils.data import Dataset, IterableDataset
import pandas as pd
import json
import random
import os

class Pretrain(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, args, length=None):
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
                else:
                    raise Exception('The given dataset does not exist in data directory.')
        else:
            if self.args.dataset=='IL':
                self.dataset = pd.read_csv('data/IL.csv')
            elif self.args.dataset=='IL_template':
                self.dataset = pd.read_csv('data/IL_template.csv')
            elif self.args.dataset=='IL_notemplate':
                self.dataset = pd.read_csv('data/IL_notemplate.csv')
            else:
                raise Exception('The given dataset does not exist in data directory.')
        
        print(f'Length of dataset retrieving is.. {len(self.dataset)}')
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        # continual pretraining
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
    def __init__(self, tokenizer, input_length, output_length, args):
        self.args = args
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.output_length = output_length

        self.data_lst = []
        # Getting the dirs of data chunks
        lst = os.listdir(self.args.dataset)
        lst.sort()
        for l in lst:
            self.data_lst.append(self.args.dataset+'/'+l)
        
        self.data_index = 0
        self.dataset = pd.read_csv(self.data_lst[self.data_index])
        self.data_index_limit = len(self.dataset)
        self.data_index_total = 0
        
    def __len__(self):
        return self.args.len_data
    
    def get_new_chunk(self):
        self.data_index +=1
        self.dataset = pd.read_csv(self.data_lst[self.data_index])
        self.data_index_total += self.data_index_limit
        self.data_index_limit = len(self.dataset)
        return 0

    def convert_to_features(self, example_batch, index=None):
        # continual pretraining
        input_ = example_batch['input']
        target_ = example_batch['output']
        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")                          
        return source, targets
  
    def __getitem__(self, index):
        index = index - self.data_index_total  
        if index == self.data_index_limit:
            index = self.get_new_chunk()
            
        source, targets = self.convert_to_features(self.dataset.iloc[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}