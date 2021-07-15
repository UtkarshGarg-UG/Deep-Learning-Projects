import numpy as np
#import torch
#from torch import nn


def train(n_epochs, loader, model, optimizer, criterion, use_cuda):
    """returns trained model"""
    losses = []
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        
        
        ###################
        # train the model #
        ###################
        # set the module to training mode
        model.train()
        
        
        for batch_idx, (data, target) in enumerate(loader):
            data = data.view(data.shape[0], -1)
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            
            preds = model(data)
            loss = criterion(preds, target)
            loss.backward()
        
            optimizer.step()
            optimizer.zero_grad()

            train_loss+= ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
            losses.append(loss.data.item())
            #if batch_idx>=max_batch_to_train:
            #    break
        print('Epoch {} losses : {}'.format(epoch, train_loss))
        
            
    
        
        
    return losses

def validation(loader, model, criterion, use_cuda):
    ######################    
        # validate the model #
    ######################
    # set the model to evaluation mode
    valid_loss = 0.0
    correct=0
    total = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(loader):
        data = data.view(data.shape[0], -1)
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        preds = model(data)
        loss = criterion(preds, target)

        valid_loss+= ((1 / (batch_idx + 1)) * (loss.data.item() - valid_loss))
        
        # convert output probabilities to predicted class
        pred = preds.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
        
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
    print('Valid Loss: ', valid_loss)




    return valid_loss

    