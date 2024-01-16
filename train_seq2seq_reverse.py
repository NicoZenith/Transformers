import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformer_layers import Seq2Seq
import numpy as np 
from utils import * 
import os
from shared_variables import * 


dir_files = './results/'+outf
dir_checkpoints = './checkpoints/'
os.makedirs(dir_checkpoints, exist_ok=True)


# Instantiate the dataset
train_dataset = SimpleSeq2SeqDataset(max_sequence_length=max_len, num_samples=10000, vocab_size=vocab_size, eos_token=vocab_size-1)
val_dataset = SimpleSeq2SeqDataset(max_sequence_length=max_len, num_samples=1000, vocab_size=vocab_size, eos_token=vocab_size-1)


batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Seq2Seq(d_model, dropout, max_len, vocab_size, vocab_size, num_heads, num_layers, device)
model.to(device)

num_epochs = 200
learning_rate = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

train_losses = []
val_losses = []

show_every = 20

# print(input_seqs[0])
# print(model.generate(input_seqs[0].unsqueeze(0), input_seq_lengths[0].unsqueeze(0), max_len, vocab_size-1, 1) )


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
print(input_seqs[batch_point])
print(target_seqs)
print(model.generate(input_seqs[0].unsqueeze(0), input_seq_lengths[0].unsqueeze(0), max_len, vocab_size-1, 1) )



