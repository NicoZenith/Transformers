import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformer_layers import Seq2Seq
import numpy as np 

class SimpleSeq2SeqDataset(Dataset):
    def __init__(self, max_sequence_length=10, num_samples=1000, vocab_size=1000, eos_token=999, start_token=1):
        self.max_sequence_length = max_sequence_length
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.eos_token = eos_token
        self.start_token = start_token
        self.data = self.generate_data()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        input_seq, input_seq_length = self.pad_sequence(input_seq, start=False)
        label_seq, _ = self.pad_sequence(target_seq, start=False)
        target_seq, target_seq_length = self.pad_sequence(target_seq, start=True)
        

        return {
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'target_seq': torch.tensor(target_seq, dtype=torch.long),
            'input_seq_length': input_seq_length,
            'target_seq_length': target_seq_length,
            'label_seq' : label_seq
        }

    def generate_data(self):
        data = []
        for _ in range(self.num_samples):
            sequence_length = torch.randint(1, self.max_sequence_length -1, (1,)).item()
            sequence = torch.randint(2, self.vocab_size - 2, (sequence_length,))  # Use vocab_size - 1 to exclude EOS token and exclude start token 
            reversed_sequence = torch.flip(sequence, [0])
            data.append((sequence, reversed_sequence))
        return data

    def pad_sequence(self, sequence, start=False):
        original_length = len(sequence) + 1
        # Pad or truncate the sequence to the specified max_sequence_length
        if len(sequence) < self.max_sequence_length:
            padding_length = self.max_sequence_length - len(sequence)
            if start:
                sequence = torch.cat( (torch.tensor([self.start_token]), sequence, torch.tensor([self.eos_token]), torch.zeros(padding_length - 2, dtype=sequence.dtype) ) )
            else:
                sequence = torch.cat((sequence, torch.tensor([self.eos_token]), torch.zeros(padding_length - 1,  dtype=sequence.dtype)))
        
        return sequence, original_length


batch_size = 64
max_len = 10
num_layers = 4
num_heads = 4
dropout = 0.2
d_model = 384
num_epochs = 100
learning_rate = 1e-4
show_every = 100


# Instantiate the dataset
dataset = SimpleSeq2SeqDataset(max_sequence_length=10, num_samples=100, vocab_size=100, eos_token=99)

# Create a DataLoader with a custom collate_fn (no need to pad in collate_fn now)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Example of how to use the DataLoader
batch = next(iter(dataloader))
input_seqs = batch['input_seq']
target_seqs = batch['target_seq']
input_seq_lengths = batch['input_seq_length']
target_seq_lengths = batch['target_seq_length']
label_seqs = batch['label_seq']


# Your training loop or model training code here
print(input_seqs[0])
print(target_seqs[0])
print(label_seqs[0])
print(input_seq_lengths)
print(target_seq_lengths)



device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Seq2Seq(d_model, dropout, max_len, 1000, num_heads, num_layers, device)
model.to(device)

print(input_seqs[0].unsqueeze(0).shape)


print(model.generate(input_seqs[0].unsqueeze(0), input_seq_lengths[0].unsqueeze(0), 10, 99, 1, forward_mask=True) )



optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

train_losses = []

show_every = 20


for e in range(num_epochs):
    store_train_losses = []
    for i, batch in enumerate(dataloader):
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


        if i%show_every == 0:
            print("Training loss {}".format(np.mean(store_train_losses[-show_every:])))



    train_losses.append(np.mean(store_train_losses))

    # print("Epoch {}: Train Loss: {:.4f}, Validation Loss: {:.4f}".format(e, train_losses[-1], val_losses[-1]))



print(input_seqs[0])
print(model.generate(input_seqs[0].unsqueeze(0), input_seq_lengths[0].unsqueeze(0), 10, 99, 1, forward_mask=True) )