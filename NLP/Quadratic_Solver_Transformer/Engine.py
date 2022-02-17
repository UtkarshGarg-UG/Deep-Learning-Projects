import torch
from tqdm import tqdm
def train(model, iterator, optimizer, criterion, clip, device, num_train_steps):
    
    model.train()
    
    epoch_loss = 0
    
    train_loss=0
    train_sent_acc = 0
    for batch_idx, batch in tqdm(enumerate(iterator), total = num_train_steps):
        
        src = batch[0].to(device)
        trg = batch[1].to(device)
        tar_len = len(trg[0])
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        
        #sentence accuracy
        preds = torch.argmax(output, dim=1)
        preds_sents = preds.reshape(src.shape[0], tar_len-1)
        #print(preds_sents[0])
        target_sents = trg.reshape(src.shape[0], tar_len-1)
        #print(target_sents[0])
        sent_acc = sum(torch.all(torch.eq(preds_sents,target_sents),axis=1)).item()/src.shape[0]
        
        
        train_sent_acc += ((1 / (batch_idx + 1)) * (sent_acc - train_sent_acc))
        train_loss+= ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
        epoch_loss += loss.item()
        
        #if batch_idx%500==0:
        #    print('\nAvg train loss for last {} steps: {:.5f}'.format(batch_idx, train_loss))
        #    print('Avg train sent accuracy for last {} steps: {:.2f}'.format(batch_idx, train_sent_acc))
        
    return epoch_loss / len(iterator), train_sent_acc



def evaluate(model, iterator, criterion, device, num_val_steps):
    
    model.eval()
    
    epoch_loss = 0
    val_sent_acc = 0
    with torch.no_grad():
    
        for batch_idx, batch in tqdm(enumerate(iterator), total = num_val_steps):

            src = batch[0].to(device)
            trg = batch[1].to(device)
            tar_len = len(trg[0])
            output, _ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
            
            #sentence accuracy
            preds = torch.argmax(output, dim=1)
            preds_sents = preds.reshape(src.shape[0], tar_len-1)
            #print(preds_sents[0])
            target_sents = trg.reshape(src.shape[0], tar_len-1)
            #print(target_sents[0])
            sent_acc = sum(torch.all(torch.eq(preds_sents,target_sents),axis=1)).item()/src.shape[0]
        
        
            val_sent_acc += ((1 / (batch_idx + 1)) * (sent_acc - val_sent_acc))
        
    return epoch_loss / len(iterator), val_sent_acc