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
from torchtext.vocab import vocab 
from collections import Counter


# Define tokenizers for English and French
english_tokenizer = get_tokenizer("spacy", language='en_core_web_sm')
french_tokenizer = get_tokenizer("spacy", language='fr_core_news_sm')

outf = "translation"

dir_files = './results/'+outf
dir_checkpoints = './checkpoints/'
os.makedirs(dir_checkpoints, exist_ok=True)


def load_tatoeba_data(file_path, source_column, target_column):
    df = pd.read_csv(file_path, delimiter='\t', header=None)
    # df = df.head(20)
    source_data = df[source_column].tolist()
    target_data = df[target_column].tolist()
    return source_data, target_data

def tokenize_sentences(sentence_pairs, max_sequence_length=50):
    tokenized_pairs = []
    for source, target in sentence_pairs:
        source_tokens = english_tokenizer(source)
        target_tokens = french_tokenizer(target)
        if len(source_tokens) < max_sequence_length and len(target_tokens) < max_sequence_length:
            tokenized_pairs.append((source_tokens, target_tokens))
    return tokenized_pairs

def create_sentence_pairs(file_path, source_column, target_column):
    source_data, target_data = load_tatoeba_data(file_path, source_column, target_column)
    sentence_pairs = list(zip(source_data, target_data))
    tokenized_pairs = tokenize_sentences(sentence_pairs)
    return tokenized_pairs

class TatoebaDataset(Dataset):
    def __init__(self, tokenized_pairs, vocab_eng, vocab_fr):
        self.tokenized_pairs = tokenized_pairs
        self.vocab_eng = vocab_eng
        self.vocab_fr = vocab_fr 

    def __len__(self):
        return len(self.tokenized_pairs)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.tokenized_pairs[idx]

        input_seq = torch.tensor([self.vocab_eng[token] for token in input_seq])
        input_seq = torch.cat((input_seq, torch.tensor([self.vocab_eng['<eos>']])))

        target_seq = torch.tensor([self.vocab_fr[token] for token in target_seq])

        label_seq = torch.cat((target_seq, torch.tensor([self.vocab_fr['<eos>']])))

        target_seq = torch.cat((torch.tensor([self.vocab_fr['<bos>']]), target_seq))
        input_seq_length = len(input_seq)
        target_seq_length = len(target_seq)
        
        return {
            'input_seq': input_seq,
            'target_seq': target_seq,
            'label_seq' : label_seq,
            'input_seq_length': input_seq_length,
            'target_seq_length': target_seq_length
        }


# Example usage:
file_path = 'tatoeba_data.tsv'  # Replace with the actual file path
source_column = 1  # Adjust these indices according to your data
target_column = 3

tokenized_pairs = create_sentence_pairs(file_path, source_column, target_column)

n = int(0.9*len(tokenized_pairs)) # first 90% will be train, rest val
train_tokenized_pairs = tokenized_pairs[:n]
val_tokenized_pairs = tokenized_pairs[n:]


def build_vocabs(tokenized_pairs):
    # Calculate word frequencies and create a Counter object
    word_counter_eng = Counter(word for pair in tokenized_pairs for word in pair[0])
    word_counter_fr = Counter(word for pair in tokenized_pairs for word in pair[1])
    vocab_eng = vocab(word_counter_eng, specials = ['<pad>', '<bos>', '<eos>', '<unk>'])
    vocab_fr = vocab(word_counter_fr, specials = ['<pad>', '<bos>', '<eos>', '<unk>'])
    
    return vocab_eng, vocab_fr

vocab_eng, vocab_fr = build_vocabs(tokenized_pairs)



def encode(sentence, vocab):
    return [vocab[token] if token in vocab else vocab['<unk>'] for token in sentence]

def decode(encoded_sequence, vocab):
    itos = vocab.get_itos()
    return [itos[token] for token in encoded_sequence]


# Create the dataset with tokenization
train_dataset = TatoebaDataset(train_tokenized_pairs, vocab_eng, vocab_fr)
val_dataset = TatoebaDataset(val_tokenized_pairs, vocab_eng, vocab_fr)


# Create a DataLoader with batch_size and collate_fn for padding
batch_size = 32  # Adjust as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

batch = next(iter(train_dataloader))


device = 'cuda' if torch.cuda.is_available() else 'cpu'

d_model = 384
dropout = 0.2
max_len = 50
vocab_size_eng = len(vocab_eng)
vocab_size_fr = len(vocab_fr)
num_heads = 6 
num_layers = 6

model = Seq2Seq(d_model, dropout, max_len, vocab_size_eng, vocab_size_fr, num_heads, num_layers, device)
model.to(device)

num_epochs = 100
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

    # Save the model checkpoint after each epoch
    checkpoint_filename = os.path.join(dir_checkpoints, outf+'_model.pth')
    torch.save(model.state_dict(), checkpoint_filename)



batch_point = 10
print("Input sequence : ")
print(decode(input_seqs[batch_point], vocab_eng))
print("Target sequence : ")
print(decode(target_seqs[batch_point], vocab_fr))
print("Generated sequence : ")
print(decode(model.generate(input_seqs, input_seq_lengths, max_len, vocab_fr['<eos>'], vocab_fr['<bos>'])[batch_point], vocab_fr) )





