import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset



class Vocabulary:
    def __init__(self, freq_threshold, max_size):
        #defining pad, start of sentence, end of sentence and unknown token index
        self.itos = {0:"<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"} #index to string dict
        self.stoi = {k:j for j,k in self.itos.items()} #string to index dict
        self.freq_threshold = freq_threshold #minimum word frequency needed to be included in vocab
        self.max_size = max_size #max vocab size
        
    def __len__(self):
        return len(self.itos)
    
    
    
    '''
    a simple tokenizer that splits on space and converts the sentence to list of words 
    '''
    @staticmethod #static method is independent of a class instance
    def tokenizer(text):
        return [tok.lower().strip() for tok in text.split(' ')]
    
    '''
    build the vocab
    '''
    def build_vocabulary(self,sentence_list):
        frequencies = {}
        idx = 4
        
        #calculate freq of words
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies.keys():
                    frequencies[word]=1
                else:
                    frequencies[word]+=1
        
        #limit vocab by removing low freq words
        frequencies = {k:v for k,v in frequencies.items() if v>self.freq_threshold}
        #limit vocab to the max size specified
        if len(frequencies)>self.max_size-4:
            frequencies = dict(sorted(frequencies.items(), key = lambda x: -x[1])[:self.max_size-4]) #-4 for start,end, pad, unk token
        #create vocab
        for word in frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx+=1
                
    '''
    convert the list of words to a list of corresponding indexes
    '''
    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            else:
                numericalized_text.append(self.stoi['<UNK>'])
        
        return numericalized_text
                    

class Train_Dataset(Dataset): 
    
    '''
    Initiating Variables
    source_column : the name of source text column in the dataframe
    target_columns : the name of target text column in the dataframe
    transform : If we want to add any augmentation
    freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
    source_vocab_max_size : max source vocab size
    target_vocab_max_size : max target vocab size
    '''
    def __init__(self, df,source_column, target_column, transform = None, freq_threshold = 5, 
                 source_vocab_max_size= 10000, target_vocab_max_size = 10000):
        self.df = df
        self.transform = transform
        
        #get source and target texts
        self.source_texts = self.df[source_column]
        self.target_texts = self.df[target_column]
        
        ##VOCABULARY class will be created below
        #create source vocab
        self.source_vocab = Vocabulary(freq_threshold,source_vocab_max_size)
        self.source_vocab.build_vocabulary(self.source_texts.tolist())
        #create target vocab
        self.target_vocab = Vocabulary(freq_threshold,target_vocab_max_size)
        self.target_vocab.build_vocabulary(self.target_texts.tolist())
        
        
        
    def __len__(self):
        return len(self.df)
    
    #getitem gets 1 example at a time. This is done before a batch is created
    def __getitem__(self, index):
        source_text = self.source_texts[index]
        #print(source_text)
        target_text = self.target_texts[index]
        #print(target_text)
        if self.transform is not None:
            source_text = self.transform(source_text)
            
        #numericalize texts ['<SOS>','cat', 'in', 'a', 'bag','<EOS>'] -> [1,12,2,9,24,2]
        numerialized_source = [self.source_vocab.stoi["<SOS>"]]
        numerialized_source += self.source_vocab.numericalize(source_text)
        numerialized_source.append(self.source_vocab.stoi["<EOS>"])
    
        numerialized_target = [self.target_vocab.stoi["<SOS>"]]
        numerialized_target += self.target_vocab.numericalize(target_text)
        numerialized_target.append(self.target_vocab.stoi["<EOS>"])
        #print(numerialized_source)
        return torch.tensor(numerialized_source), torch.tensor(numerialized_target) 
    

"""
Create a separate dataset for validation as we
are going to reuse train's vocabulary.
Better practice to define a separate class for validation
""" 

class Validation_Dataset:
    def __init__(self, train_dataset, df, source_column, target_column, transform = None):
        self.df = df
        self.transform = transform
        
        #train dataset will be used as lookup for vocab
        self.train_dataset = train_dataset
        
        #get source and target texts
        self.source_texts = self.df[source_column]
        self.target_texts = self.df[target_column]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
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



"""
a class to create padding to the batches
 collat_fn in dataloader is used for postprocessing on a single batch. Like __getitem__ in dataset class was used
 for a single example
"""

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    # __call__ :
    ##    First the obj is created using MyCollate(pad_idx).
    ##    Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        source = [item[0] for item in batch]
        source = pad_sequence(source, batch_first=False, padding_value = self.pad_idx)
        target = [item[1] for item in batch]
        target = pad_sequence(target, batch_first=False, padding_value = self.pad_idx)
        
        return source,target


# If we run a next(iter(data_loader)) we get an output of batch_size * (num_workers+1)
def get_train_loader(dataset, batch_size, num_workers=1, shuffle=True, pin_memory=False):
    pad_idx = dataset.source_vocab.stoi['<PAD>']
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,
                       pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx))
    return loader

def get_valid_loader(dataset, train_dataset, batch_size, num_workers=1, shuffle=True, pin_memory=False):
    pad_idx = train_dataset.source_vocab.stoi['<PAD>']
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,
                       pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx))
    return loader