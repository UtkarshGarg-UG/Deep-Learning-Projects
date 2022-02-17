#########################################
#       IMPORT LIBRARIES
#########################################
import pandas as pd
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import random
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
#from torch.utils.data import random_split, DataLoader
import math
from torch import nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchmetrics
import wandb
import re
import argparse
import json
##################################################
#   Import classes for Data Module
##################################################

from data_layer import Vocabulary, Train_Dataset, Validation_Dataset, MyCollate, DataModule

###############################
#   Import classes for Model
################################

from model import Encoder, Decoder, Seq2Seq

###############################################
#   Import Methods for Training and Evaluation
###############################################

from Engine import train, evaluate

#########################################
#             Set Seeds
#########################################


def set_seed(seed):
    # Setting the seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)

def count_parameters(model):
    """
    desc :
        get the number of weights in model
    params: 
        model
        
    returns:
        number of weights in model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    
    """
    desc:
        Initialize weights of the model
        We initialize the weights using xavier uniform init
        
    params:
        m: weight
        
    """
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def get_optimizer(model, learning_rate):
    """
    desc:
        define optimizer
        
    params:
        model
        learning_rate
       
    returns:
        the optimizer
    """
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    return optimizer
            

def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs   
    

def checkpoint_builder(model, optimizer, criterion, epoch, model_params, data_module, checkpoint_path):
    """
    desc:
        creates checkpoint for the model
    params:
        model: trained model
        optimizer
        criterion: loss fn
        epoch: checkpoint epoch
        model_params: hyperparams of model
        data_module: object of DataModule => contains the datasets and vocabulary
        checkpoint_path: path to save checkpoint. Include checkpoint name
    """
    
    torch.save({'model_state_dict': model.state_dict(),
                        'optim_state_dict':optimizer.state_dict(),
                        'criterion': criterion,
                        'epoch': epoch, 
                        'model_params': model_params,
                        'source_stoi': data_module.train_dataset.source_vocab.stoi,
                        'source_itos': data_module.train_dataset.source_vocab.itos,
                        'target_stoi': data_module.train_dataset.target_vocab.stoi,
                        'target_itos': data_module.train_dataset.target_vocab.itos}, checkpoint_path)

    



def main(device, train_data_path, model_params, source_column = 'factors', 
             target_column = 'expressions', batch_size = 512, val_frac = 0.1, 
                 learning_rate = 0.0005, n_epochs = 10, early_stopping_limit = 3, 
                     checkpoint_path = 'checkpoints/checkpoint.pt', resume_from_checkpoint = False):
    

    """
    desc:
        This is the main func. It does the following:
            * We create the data module (dataset, loaders, vocabulary)
            * Define the model
            * Model training with option of resuming from checkpoint
            * The main training loop
            * Checkpoint save
            
    params:
        device: str: cuda/cpu
        train_data_path: str: path of the training file (we assume a txt with "source = text")
        model_params: dict: hyperparameters for the model
        source_column: str: the source column
        target_columns: str: the target column
        batch_size: int: batch size
        val_frac: float: validation data ratio
        learning_rate: float: learning rate for the optimizer
        n_epochs: int: number of epochs
        early_stopping_limit: int: wait for these many epochs before early stopping due to non-reduction of val loss
        checkpoint_path: str: path to save/load the checkpoint 
        resume_from_checkpoint: bool: if True, it resumes the training from checkpoint at checkpoint_path
        
    return:
        None
    """
    
    
    #################
    # Create Data Module - Datasets, Vocab and Data Iterator
    #################
    print('\nCreating data module...')
    data_module = DataModule(train_data_path, source_column, 
                             target_column, batch_size)
    
    print(f'Using split ratio of {val_frac} for train val split')
    data_module.setup(val_frac = val_frac)

    train_iterator = data_module.train_dataloader()
    valid_iterator = data_module.val_dataloader()
    
    print('Data module created.')
    #to be used for engine's tqdm
    num_train_steps = data_module.train_len // batch_size
    num_val_steps = data_module.val_len // batch_size

    ###############
    # Create Model
    ###############
    
    print('\nInitiating Model..')
    enc = Encoder(model_params['INPUT_DIM'], 
                  model_params['HID_DIM'], 
                  model_params['ENC_LAYERS'], 
                  model_params['ENC_HEADS'], 
                  model_params['ENC_PF_DIM'], 
                  model_params['ENC_DROPOUT'], 
                  device)

    dec = Decoder(model_params['OUTPUT_DIM'], 
                  model_params['HID_DIM'], 
                  model_params['DEC_LAYERS'], 
                  model_params['DEC_HEADS'], 
                  model_params['DEC_PF_DIM'], 
                  model_params['DEC_DROPOUT'], 
                  device)


    SRC_PAD_IDX = data_module.train_dataset.source_vocab.stoi['<PAD>']
    TRG_PAD_IDX = data_module.train_dataset.target_vocab.stoi['<PAD>']

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = get_optimizer(model, learning_rate)
    
    #define criterion
    criterion = nn.CrossEntropyLoss()
    
    #train from checkpoint
    if resume_from_checkpoint:
        print('Loading Checkpoint ...')
        checkpoint = torch.load(checkpoint_path)
        print('Setting Models state to checkpoint...')
        assert checkpoint['model_params'] == model_params
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Model state set.')
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        print('Optimizer state set.')
        
    #train from scratch
    else:
        model.apply(initialize_weights)
    

   
    ########################
    # Training Loop
    ########################
    N_EPOCHS = n_epochs
    CLIP = 1
    early_stopping_counter = 0

    best_valid_loss = float('inf')
    print('------------------------------------------\nStarting Training...')
    print(f'Early stopping counter is: {early_stopping_limit}')
    
    for epoch in range(N_EPOCHS):
        print('###################################')
        print(f'    Starting Epoch => {epoch}')
        print('###################################')
        start_time = time.time()
        
        #Engine => train and validate on batches in epoch
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP, device, num_train_steps)
        valid_loss, val_sent_acc = evaluate(model, valid_iterator, criterion, device, num_val_steps)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        #early stopping, checkpointing
        if valid_loss < best_valid_loss:
            early_stopping_counter = 0
            best_valid_loss = valid_loss
            #create checkpoint
            checkpoint_builder(model, optimizer, criterion, epoch, model_params, data_module, checkpoint_path)
        
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_limit:
                print(f'\nLoss did not reduce for last {early_stopping_counter} epochs. Stopped training..')
                break

        print(f'\nEpoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.5f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.5f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print(f'\t Val. Sent Accuracy: {val_sent_acc:.3f}')
        
        
def init_argparse():
    
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--seed', default=1, type=int, help='random seed')
    arg_parser.add_argument('--train_path', default='train.txt', type=str, help='raw train file')
    arg_parser.add_argument('--cuda', default=True, type=bool, help='use cuda or not')
    arg_parser.add_argument('--gpu_name', default=0, type=int, help='which gpu to use')
    arg_parser.add_argument('--bs', default=512, type=int, help='batch size')
    arg_parser.add_argument('--val_frac', default=0.1, type=float, help='validation ratio')
    arg_parser.add_argument('--lr', default=0.0005, type=int, help='learning rate')
    arg_parser.add_argument('--epochs', default=150, type=int, help='num epochs')
    arg_parser.add_argument('--early_stop_count', default=5, type=int, help='early stopping count')
    arg_parser.add_argument('--checkpoint_path', default='checkpoints/checkpoint.pt', type=str, help='checkpoint path')
    arg_parser.add_argument('--resume_training', default=False, type=bool, help='resume training from checkpoint')

    return (arg_parser.parse_args())
    
        
if __name__ == "__main__":
    
    args = init_argparse()
    
    if os.path.isfile(args.checkpoint_path):
        print('!!!!!!!!!!!!!!!!!!!!!!!!')
        print('WARNING! checkpoint already exists. \nConsider using another name to save new checkpoint')
        print('!!!!!!!!!!!!!!!!!!!!!!!!\n')
    print(f'Training will run with following arguments: {args}')

    #set seed
    set_seed(args.seed)
    
    model_params = json.load(open('model_config.json'))
    
    cuda = args.cuda
    
    if cuda:
        device = torch.device(args.gpu_name if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    
    print(f"\nTraining will start on {device}.")
    
    
    
    main(
            device = device, 
            train_data_path = args.train_path, 
            model_params = model_params, 
             source_column = 'factors',     
            target_column = 'expressions', 
            batch_size = args.bs, 
            val_frac = args.val_frac, 
            learning_rate = args.lr, 
            n_epochs = args.epochs, 
            early_stopping_limit = args.early_stop_count, 
            checkpoint_path = args.checkpoint_path, 
            resume_from_checkpoint = args.resume_training
    )
    
    

    
        