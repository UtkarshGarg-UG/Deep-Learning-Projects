
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
#from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

"""
### Define Encoder
The input to encoder is the whole sentence. Therefore, the size of x in forward is (seq_length, batch_size)
"""
class Encoder(nn.Module):
    '''
    input_size -> source vocab size
    '''
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_p)
        
    def forward(self, x):
        # shape of x-> (seq_length, batch_size)
        embedding = self.dropout(self.embedding(x)) #shape-> (seq_length, batch_size, embedding_size)
        
        outputs, (hidden,cell) = self.rnn(embedding)
        
        return hidden, cell #shape-> (num_layers, batch_size, hidden_size)



"""
### Define Decoder
* Input is one word at a time. shape of x is-> (1, batch_size)
* embedding size-> (1, batch_size, embedding_size)
* outputs-> (1, batch_size, hidden_size)
* predictions -> (batch_size, output_size)
"""

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_p):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = dropout_p)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        # for next(iter(train_loader))[1][0] will give us the first word for batch_size (,batch_size)
        # to input to model, we need to add a dimension at 0
        # x-> (, batch_size) -> convert to (1, batch_size)
        x = x.unsqueeze(0)
        #print('input unsqueezed: ', x.shape)
        embedding = self.dropout(self.embedding(x)) #(1, batch_size, embedding_size)
        #print('embedding: ', embedding.shape)
        output, (hidden, cell) = self.rnn(embedding, (hidden, cell)) #UG-> Need to understand how LSTM takes both inputs
        #print('output: ', output.shape, '\nhidden: ', hidden.shape)
        #outputshape -> (1, batch_size, hidden_size)
        predictions = self.fc(output) #(1, batch_size, output_size)
        #print('predictions: ', predictions.shape)
        #to send predictions to loss func we need to remove 0th dim
        predictions = predictions.squeeze(0)
        #print('predictions squeezed: ', predictions.shape)
        return predictions, hidden, cell
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_size, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_size = output_size
        self.device = device
        
    def forward(self, source , target, teacher_force_ratio = 0.5):
        #source-> (max_seq_length, batch_size)
        #target-> (max_seq_length, batch_size)
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.output_size #the size of target vocab
        
        #this tensor will be used for predicting output one by one
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(source)
        
        #get first token from input to the decoder -> <SOS>
        x = target[0] #shape -> (,batch_size)
        
        for t in range(1, target_len):
            #get output, hidden, cell from decoder. Input is x, previous hidden and cell
            prediction, hidden, cell = self.decoder(x, hidden, cell)
            #prediction shape-> (batch_size, target_vocab_size)
            
            #store next output prediction
            outputs[t] = prediction
            
            #get best word the decoder predicted (index of target_vocab)
            best_guess = prediction.argmax(1)
            #print('best guess: ', best_guess.shape)
            #print('target[t]: ', target[t].shape)
            # teacher forcing ratio is used to sometime input the previous prediction and sometime give the actual word from training data
            x = target[t] if random.random()<teacher_force_ratio else best_guess
            
            
        return outputs #shape-> (seq_length, batch_size, target_vocab)