import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformer_layers import Seq2Seq
import numpy as np 
from utils import * 
import os
from shared_variables import * 
import pandas as pd 
from torchtext.data.utils import get_tokenizer
import pickle 



# Define tokenizers for English and French


dir_files = './results/'+outf
dir_checkpoints = './checkpoints/'
os.makedirs(dir_checkpoints, exist_ok=True)


# Example usage:
file_path = 'data/tatoeba_data.tsv'  # Replace with the actual file path
source_column = 3  # Adjust these indices according to your data
target_column = 1

french_tokenizer = get_tokenizer("spacy", language='fr_core_news_sm')
english_tokenizer = get_tokenizer("spacy", language='en_core_web_sm')


# Check if tokenized pairs exist, if not, create them
if os.path.exists("fr_eng_tokenized_pairs.pkl"):
    with open("fr_eng_tokenized_pairs.pkl", "rb") as f:
        tokenized_pairs = pickle.load(f)
else:
    # Create tokenized_pairs using your function
    tokenized_pairs = create_tokenized_pairs(file_path, source_column, target_column, french_tokenizer, english_tokenizer)
    
    # Save tokenized_pairs to a file
    with open("fr_eng_tokenized_pairs.pkl", "wb") as f:
        pickle.dump(tokenized_pairs, f)

n = int(0.9*len(tokenized_pairs)) # first 90% will be train, rest val
train_tokenized_pairs = tokenized_pairs[:n]
val_tokenized_pairs = tokenized_pairs[n:]

vocab_source, vocab_target = build_vocabs(tokenized_pairs)


def encode(sentence, vocab):
    return [vocab[token] if token in vocab else vocab['<unk>'] for token in sentence]

def decode(encoded_sequence, vocab):
    itos = vocab.get_itos()
    return [itos[token] for token in encoded_sequence]


# Create the dataset with tokenization
train_dataset = TatoebaDataset(train_tokenized_pairs, vocab_source, vocab_target)
val_dataset = TatoebaDataset(val_tokenized_pairs, vocab_source, vocab_target)


# Create a DataLoader with batch_size and collate_fn for padding
batch_size = 32  # Adjust as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

batch = next(iter(train_dataloader))


device = 'cuda' if torch.cuda.is_available() else 'cpu'


vocab_size_source = len(vocab_source)
vocab_size_target = len(vocab_target)


model = Seq2Seq(d_model, dropout, max_len, vocab_size_source, vocab_size_target, num_heads, num_layers, device)
model.to(device)

num_epochs = 50
learning_rate = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

train_losses = []
val_losses = []

show_every = 20

for e in range(num_epochs):
    store_train_losses = []
    store_val_losses = []
    for i, batch in enumerate(train_dataloader):
        input_seqs = batch['input_seq'].to(device)
        target_seqs = batch['target_seq'].to(device)
        labels_seqs = batch['label_seq'].to(device)
        input_seq_lengths = batch['input_seq_length']
        target_seq_lengths = batch['target_seq_length']
        model.zero_grad()

        logits = model(input_seqs, target_seqs, input_seq_lengths, target_seq_lengths)

        loss = model.compute_loss(logits, labels_seqs, target_seq_lengths)
        loss.backward()
        optimizer.step()

        store_train_losses.append(loss.item())



        # if i%show_every == 0:
        #     print("Training loss {}".format(np.mean(store_train_losses[-show_every:])))

    
    for i, batch in enumerate(val_dataloader):
        input_seqs = batch['input_seq'].to(device)
        target_seqs = batch['target_seq'].to(device)
        labels_seqs = batch['label_seq'].to(device)
        input_seq_lengths = batch['input_seq_length']
        target_seq_lengths = batch['target_seq_length']
        with torch.no_grad():
            logits = model(input_seqs, target_seqs, input_seq_lengths, target_seq_lengths)

        loss = model.compute_loss(logits, labels_seqs, target_seq_lengths)

        store_val_losses.append(loss.item())




    train_losses.append(np.mean(store_train_losses))
    val_losses.append(np.mean(store_val_losses))
    

    print("Epoch {}: Train Loss: {:.4f}, Validation Loss: {:.4f}".format(e, train_losses[-1], val_losses[-1]))

    save_fig_trainval(e, train_losses, val_losses, dir_files)

    torch.save(
        {'train_losses': train_losses,
        'val_losses': val_losses
        }, dir_files + '_losses.pth'
    )
    print(f'Losses successfully saved.')


    # Save the model checkpoint after each epoch
    checkpoint_filename = os.path.join(dir_checkpoints, outf+'_model.pth')
    torch.save(model.state_dict(), checkpoint_filename)



batch_point = 10
print("Input sequence : ")
print(decode(input_seqs[batch_point], vocab_source))
print("Target sequence : ")
print(decode(target_seqs[batch_point], vocab_target))
print("Generated sequence : ")
print(decode(model.generate(input_seqs, input_seq_lengths, max_len, vocab_source['<eos>'], vocab_target['<bos>'])[batch_point], vocab_target) )





