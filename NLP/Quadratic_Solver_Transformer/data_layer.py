import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

import pytorch_lightning as pl

from utils import tokenize


#get number of CPUs for setting num workers in dataloader
import multiprocessing

NUM_WORKERS = multiprocessing.cpu_count()


class Vocabulary:
    def __init__(self, freq_threshold, max_size):
        '''
        desc : Initialize Vocab hyperparameters
        params:
            freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
            max_size : max source vocab size
        
        '''
        #initiate the index to token dict
        self.itos = {0: '<PAD>', 1:'<SOS>', 2:'<EOS>', 3: '<UNK>'}
        #initiate the token to index dict
        self.stoi = {k:j for j,k in self.itos.items()} 
        
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        
        
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer(text):
        '''
        desc: Tokenize text on the basis of tokenize method
        params:
            text : sentence to be tokenized
        returns:
            tokenized text
        '''
        return tokenize(text)
    
    
    
    def build_vocabulary(self, sentence_list):
        '''
        desc: build the vocab for given sentence list
                creates stoi => string to index dictionary and
                        itos => index to string dictionary
        params:
            sentence_list: List of sentences

        '''
        frequencies = {}
        idx = len(self.itos)  #will start the vocabulary from this index {0: '<PAD>', 1:'<SOS>', 2:'<EOS>', 3: '<UNK>'}
        
        #calculate freq of words
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies.keys():
                    frequencies[word]=1
                else:
                    frequencies[word]+=1
                    
                    
        #limit vocab by removing low freq words
        frequencies = {k:v for k,v in frequencies.items() if v>self.freq_threshold} 
        
        #limit vocab to the max_size specified
        if len(frequencies)>self.max_size-idx: 
            frequencies = dict(sorted(frequencies.items(), key = lambda x: -x[1])[:self.max_size-idx]) # idx =4 for pad, start, end , unk
            
        #create vocab
        for word in frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx+=1
        
       
    def numericalize(self, text):
        '''
        desc: convert a sentence to corresponding tokens and 
                then to a list of numbers basis the stoi dictionary
        params:
            text: a single instance of a sentence
        return:
            numericalized_text : text converted to a list of numbers
        ''' 
        tokenized_text = self.tokenizer(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            else:
                numericalized_text.append(self.stoi['<UNK>'])
                
        return numericalized_text
        
            
class Train_Dataset(Dataset):
    
    def __init__(self, df, source_column, target_column, transform=None, freq_threshold = 0,
                source_vocab_max_size = 10000, target_vocab_max_size = 10000):

        '''
        desc: creates pytorch dataset for training data

        params:
        
            source_column : the name of source text column in the dataframe
            target_columns : the name of target text column in the dataframe
            transform : augmentation method
            freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
            source_vocab_max_size : max source vocab size
            target_vocab_max_size : max target vocab size

        '''

        self.df = df
        self.transform = transform
        
        #get source and target texts
        self.source_texts = self.df[source_column]
        self.target_texts = self.df[target_column]
        
        
        ##VOCAB class has been created above
        #create source vocab
        self.source_vocab = Vocabulary(freq_threshold, source_vocab_max_size)
        self.source_vocab.build_vocabulary(self.source_texts.tolist())
        #create target vocab
        self.target_vocab = Vocabulary(freq_threshold, target_vocab_max_size)
        self.target_vocab.build_vocabulary(self.target_texts.tolist())
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        '''
        desc: this function is used by the data loader by sending in an index of a row
                in data and converting it into numericalized tensors of source and target
        params:
            index: index of the data point in the dataset object

        returns:
            tensors of numericalized text of both source and target
        '''

        source_text = self.source_texts[index]
        target_text = self.target_texts[index]
        
        if self.transform is not None:
            source_text = self.transform(source_text)
            
        #numericalize texts ['<SOS>','cat', 'in', 'a', 'bag','<EOS>'] -> [1,12,2,9,24,2]
        numerialized_source = [self.source_vocab.stoi["<SOS>"]]
        numerialized_source += self.source_vocab.numericalize(source_text)
        numerialized_source.append(self.source_vocab.stoi["<EOS>"])
    
        numerialized_target = [self.target_vocab.stoi["<SOS>"]]
        numerialized_target += self.target_vocab.numericalize(target_text)
        numerialized_target.append(self.target_vocab.stoi["<EOS>"])
        return torch.tensor(numerialized_source), torch.tensor(numerialized_target) 
    



class Validation_Dataset:
     
    def __init__(self, train_dataset, valid_df, source_column, target_column, transform = None):
        """
        desc: 
            Create a separate dataset for validation as we
            are going to reuse train's vocabulary.
            Better practice to define a separate class for validation

        params:
            train_dataset : the dataset object of train. use it to get stoi and itos (vocab indexes)
            valid_df : validation pandas dataframe
            source_column : the name of source text column in the dataframe
            target_columns : the name of target text column in the dataframe
        """
        self.valid_df = valid_df
        self.transform = transform
        
        #train dataset will be used as lookup for vocab
        self.train_dataset = train_dataset
        
        #get source and target texts
        self.source_texts = self.valid_df[source_column]
        self.target_texts = self.valid_df[target_column]
    
    def __len__(self):
        return len(self.valid_df)
    
    def __getitem__(self,index):
        """
        desc: this function is used by the data loader by sending in an index of a row
                in data and converting it into numericalized tensors of source and target
        params:
            index: index of the data point in the dataset object

        returns:
            tensors of numericalized text of both source and target
        """
        source_text = self.source_texts[index]
        #print(source_text)
        target_text = self.target_texts[index]
        #print(target_text)
        if self.transform is not None:
            source_text = self.transform(source_text)
            
        #numericalize texts ['<SOS>','cat', 'in', 'a', 'bag','<EOS>'] -> [1,12,2,9,24,2]
        numerialized_source = [self.train_dataset.source_vocab.stoi["<SOS>"]]
        numerialized_source += self.train_dataset.source_vocab.numericalize(source_text)
        numerialized_source.append(self.train_dataset.source_vocab.stoi["<EOS>"])
    
        numerialized_target = [self.train_dataset.target_vocab.stoi["<SOS>"]]
        numerialized_target += self.train_dataset.target_vocab.numericalize(target_text)
        numerialized_target.append(self.train_dataset.target_vocab.stoi["<EOS>"])
        #print(numerialized_source)
        return torch.tensor(numerialized_source), torch.tensor(numerialized_target) 






class MyCollate:
    
    def __init__(self, pad_idx):
        '''
        desc: 
            class to add padding to the batches
            collat_fn in dataloader is used for post processing on a single batch. 
            Like __getitem__ in dataset class is used on single example

        params:
            pad_idx : padding index in vocabulary
        '''
        self.pad_idx = pad_idx
        
    
    #__call__:
    ##   First the obj is created using MyCollate(pad_idx) in data loader
    ##   Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        source = [item[0] for item in batch]
        source = pad_sequence(source, batch_first=True, padding_value = self.pad_idx)
        target = [item[1] for item in batch]
        target = pad_sequence(target, batch_first=True, padding_value = self.pad_idx)
        return source, target
    


    
#######################
# Data Module for Pytorch Lightning
#######################
"""
Create Data module for Pytorch Lightning
Note: Methods starting in "_" are non pytorch lighting methods
"""
class DataModule(pl.LightningDataModule):
    def __init__(self, data_path, source_column, target_column, batch_size):
        """
        desc: initialize data module by inheriting LightingDataModule
        params:
            data_path : path for the data
            source_column : the name of source text column
            target_columns : the name of target text column
            batch_size : number of datapoints in a batch
        """
        self.data_path = data_path
        self.source_column = source_column
        self.target_column = target_column
        self.batch_size = batch_size
    
    @staticmethod
    def _load_file(file_path):# -> Tuple[Tuple[str], Tuple[str]]:
        """ 
        desc: An internal helper functions that loads the file into a tuple of strings

        params: 
            file_path: path to the data file
        returns:
            factors: (LHS) inputs to the model
            expansions: (RHS) group truth
        """
        data = open(file_path, "r").readlines()
        factors, expansions = zip(*[line.strip().split("=") for line in data])
        return factors, expansions
    
    def _create_splits(self, data, val_frac):
        """ 
        desc: An internal fn used by setup for data split
        
        params: 
            val_frac: fraction of data for validation
        returns: 
            train: training pandas dataframe
            val: validation pandas dataframe
        
        """
        val_split_idx = int(len(data)*val_frac)
        data_idx = list(range(len(data)))
        np.random.shuffle(data_idx)
        val_idx, train_idx = data_idx[:val_split_idx], data_idx[val_split_idx:]
        print('len of train: ', len(train_idx))
        print('len of val: ', len(val_idx))
        train = data.iloc[train_idx].reset_index().drop('index',axis=1)
        val = data.iloc[val_idx].reset_index().drop('index',axis=1)
        self.val = val
        self.train_len = len(train)
        self.val_len = len(val)
        self.val.to_csv('val.txt',index=False, header=False, sep = '=')
        return train, val
    
    def setup(self, val_frac = 0.05, stage = None):
        '''
        desc: 
            Setup creates pytorch datasets for us
        params
            val_frac : fraction for train and validation split
            stage: fit or test
        '''
        if stage == 'fit' or stage is None:
            source, target = self._load_file(self.data_path)
            data = pd.DataFrame({self.source_column: source, self.target_column: target})#.sample(1000)
            data = data.reset_index(drop=True)
            
            #create train and validation splits
            train, val = self._create_splits(data, val_frac)
            #create datasets
            self.train_dataset = Train_Dataset(train, source_column = self.source_column, target_column = self.target_column, transform = None, freq_threshold = 0, 
                                             source_vocab_max_size= 10000, target_vocab_max_size = 10000)
            self.valid_dataset = Validation_Dataset(self.train_dataset, val, self.source_column, self.target_column)
        
        #if stage == 'test:
            #To Do
    
    
    # If we run a next(iter(data_loader)) we get an output of batch_size * (num_workers)
    def _get_train_loader(self, dataset, batch_size, num_workers=1, shuffle=True, pin_memory=False):
        '''
        desc: 
            internal fn that creates the train data loader
        params: 
            dataset : Dataset object to be used to create data loader
            batch_size : Batch size of the returned loader
            num_workers : No. of CPUs to be used 
            shuffle : shuffle train data before batching
            pin_memory : set True to speed up transfer batch from CPU to GPU

        returns:
            loader : train data loader
        '''
        pad_idx = dataset.source_vocab.stoi['<PAD>']
        loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                            shuffle=shuffle,
                           pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx))
        return loader

    def _get_valid_loader(self, dataset, train_dataset, batch_size, num_workers=1, shuffle=False, pin_memory=False):
        """
        desc: 
            internal fn that creates the valid data loader
        params:
            dataset : Dataset object to be used to create data loader
            train_dataset : train dataset to be used for getting vocab
            batch_size : Batch size of the returned loader
            num_workers : No. of CPUs to be used 
            shuffle : shuffle train data before batching
            pin_memory : set True to speed up transfer batch from CPU to GPU

        returns:
            loader : train data loader

        """
        pad_idx = train_dataset.source_vocab.stoi['<PAD>']
        loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                            shuffle=shuffle,
                           pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx))
        return loader

    #return batch of shape: (max_len, batch_size)
    def train_dataloader(self):
        '''
        desc:
            pl method to create the train data loader
        returns:
            loader : train dataloader

        '''
        loader = self._get_train_loader(self.train_dataset, self.batch_size, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True)
        return loader

    def val_dataloader(self):
        '''
        desc: 
            pl methor to create the valid data loader
        returns:
            loader : valid data loader
        '''
        loader = self._get_valid_loader(self.valid_dataset, self.train_dataset, self.batch_size, num_workers=NUM_WORKERS, pin_memory=True)
        return loader

    #def test_dataloader(self):
    #    return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers = 8)
    
    




