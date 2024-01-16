import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformer_layers import Seq2Seq
import numpy as np 
from utils import * 
import os
from shared_variables import * 
import pickle


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
val_dataset = TatoebaDataset(val_tokenized_pairs, vocab_source, vocab_target)

val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False, collate_fn=collate_batch)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab_size_source = len(vocab_source)
vocab_size_target = len(vocab_target)


model = Seq2Seq(d_model, dropout, max_len, vocab_size_source, vocab_size_target, num_heads, num_layers, device)
model.to(device)

checkpoint_filename = dir_checkpoints + outf+'_model.pth'
model.load_state_dict(torch.load(checkpoint_filename, map_location=torch.device('cpu') ) )

# Example of how to use the DataLoader
batch = next(iter(val_dataloader))
input_seqs = batch['input_seq']
target_seqs = batch['target_seq'] 
input_seq_lengths = batch['input_seq_length']
target_seq_lengths = batch['target_seq_length']
label_seqs = batch['label_seq']

def concatenate_tokens(tokens, exclude_tokens=('<bos>', '<eos>', '<pad>')):
    sentence = ' '.join(token for token in tokens if token not in exclude_tokens)
    return sentence

batch_point = 8
print("\nInput sequence : ")
input_sequence = concatenate_tokens(decode(input_seqs[batch_point], vocab_source))
print(input_sequence)
print("\nTarget sequence : ")
target_sequence = concatenate_tokens(decode(target_seqs[batch_point], vocab_target))
print(target_sequence)
print("\nGenerated sequence : ")
generated_sequence = concatenate_tokens(decode(model.generate(input_seqs, input_seq_lengths, max_len, vocab_source['<eos>'], vocab_target['<bos>'])[batch_point], vocab_target) )
print(generated_sequence)

