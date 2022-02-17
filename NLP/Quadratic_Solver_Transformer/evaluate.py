import torch
from model import Encoder, Decoder, Seq2Seq
from utils import tokenize


def load_model(checkpoint, device):

    
    model_params = checkpoint['model_params']
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

    SRC_PAD_IDX = checkpoint['source_stoi']['<PAD>']
    TRG_PAD_IDX = checkpoint['target_stoi']['<PAD>']

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model




def translate_sentence(sentence, source_stoi, source_itos, target_stoi, target_itos, model, device, attention_maps = False ,max_len = 50):
    
    model.eval()
        
    tokens = tokenize(sentence)
    
    tokens = ['<SOS>'] + tokens + ['<EOS>']
    
    src_indexes = [source_stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    
    trg_indexes = [target_stoi['<SOS>']]
    
    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)
        
        if pred_token == target_stoi['<EOS>']:
            break
    
    trg_tokens = [target_itos[i] for i in trg_indexes]
    
    if attention_maps:
        return tokens[1:-1], trg_tokens[1:-1], attention 
    else:
        return ''.join(trg_tokens[1:-1]) 