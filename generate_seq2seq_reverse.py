import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformer_layers import Seq2Seq
import numpy as np 
from utils import * 
import os
from shared_variables import * 


dir_checkpoints = './checkpoints/'

# Instantiate the dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = SimpleSeq2SeqDataset(max_sequence_length=max_len, num_samples=10000, vocab_size=vocab_size, eos_token=vocab_size-1)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = Seq2Seq(d_model, dropout, max_len, vocab_size, num_heads, num_layers, device)

checkpoint_filename = dir_checkpoints + outf+'_model.pth'
model.load_state_dict(torch.load(checkpoint_filename, map_location=torch.device('cpu') ) )

# Example of how to use the DataLoader
batch = next(iter(dataloader))
input_seqs = batch['input_seq']
target_seqs = batch['target_seq']
input_seq_lengths = batch['input_seq_length']
target_seq_lengths = batch['target_seq_length']
label_seqs = batch['label_seq']

print("Input sequence : ")
print(input_seqs)
print("Target sequence : ")
print(target_seqs)
print("Generated sequence : ")
print(model.generate(input_seqs, input_seq_lengths, max_len, vocab_size-1, 1) )

