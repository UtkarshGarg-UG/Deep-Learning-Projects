import torch

def test_on_sentence(source_text, model, train_dataset, device, max_len = 20):
    numerialized_source = [train_dataset.source_vocab.stoi["<SOS>"]]
    numerialized_source += train_dataset.source_vocab.numericalize(source_text)
    numerialized_source.append(train_dataset.source_vocab.stoi["<EOS>"])
    input_vector =  torch.tensor(numerialized_source).unsqueeze(1).to(device)

    hidden, cell = model.encoder(input_vector)
   
    #get first token from input to the decoder -> <SOS>
    x = input_vector[0] #shape -> (,batch_size)
    #outputs=torch.zeros((max_len, 1, output_size))
    #print(x)
    predicted_sentence=[]
   
    for t in range(1, max_len):
        #get output, hidden, cell from decoder. Input is x, previous hidden and cell
        prediction, hidden, cell = model.decoder(x, hidden, cell)
        #prediction shape-> (batch_size, target_vocab_size)
       
        #store next output prediction
        #outputs[t] = prediction
       
        #get best word the decoder predicted (index of target_vocab)
        best_guess = prediction.argmax(1)
        #print(best_guess)
        predicted_sentence.append(train_dataset.target_vocab.itos[best_guess.data.item()])#best_guess.data.item()

        if best_guess==2:
            break
        x = best_guess
    return ' '.join(predicted_sentence)

