
import torch 
import os
import random
import numpy as np 
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from torchtext.data.utils import get_tokenizer








def collate_batch(batch):
    """
    Collate function for custom dataset DataLoader.
    
    :param batch: List of batched data samples
    :return: Batched data as a dictionary of tensors
    """
    input_seqs = [item['input_seq'] for item in batch]
    target_seqs = [item['target_seq'] for item in batch]
    label_seqs = [item['label_seq'] for item in batch]
    input_lens = torch.tensor([item['input_seq_length'] for item in batch])
    target_lens = torch.tensor([item['target_seq_length'] for item in batch])

    # Pad sequences to the maximum sequence length in the batch
    padded_input_seqs = torch.nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=0)
    padded_target_seqs = torch.nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=0)
    padded_label_seqs = torch.nn.utils.rnn.pad_sequence(label_seqs, batch_first=True, padding_value=0)

    return {
        'input_seq': padded_input_seqs,
        'target_seq': padded_target_seqs,
        'label_seq': padded_label_seqs,
        'input_seq_length': input_lens,
        'target_seq_length': target_lens
    }



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
            'input_seq': input_seq,
            'target_seq': target_seq,
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
            if start:
                sequence = torch.cat( (torch.tensor([self.start_token]), sequence,  ) )
            else:
                sequence = torch.cat((sequence, torch.tensor([self.eos_token])))
        
        return sequence, original_length



def save_fig_trainval(epoch, train_losses, val_losses, dir_files):
    e = np.arange(0, epoch+1)
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax1.plot(e, train_losses, label='train loss')
    ax1.plot(e, val_losses, label='validation loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.set_ylim(0, 8)
    ax1.legend()
    fig.savefig(dir_files + '_trainval.pdf')





# Define a custom tokenizer that handles line breaks
def custom_tokenizer(text):
    basic_english_tokenizer = get_tokenizer('basic_english')
    tokens = []
    for paragraph in text.split('\n'):
        tokens.extend(basic_english_tokenizer(paragraph))
        # Add a token to represent line breaks
        tokens.append('<br>')
    return tokens


class GPT_Dataset(Dataset):
    def __init__(self, text, max_len):
        self.inputs = []
        self.labels = []
        for i in range(len(text) - max_len):
            x = text[i:i+max_len]
            y = text[i+1:i+max_len + 1]
            self.inputs.append(x)
            self.labels.append(y)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])




# data loading
def get_batch(data, batch_size, max_len, device):
    # generate a small batch of data of inputs x and targets y
    # data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - max_len, (batch_size,))
    x = torch.stack([data[i:i+max_len] for i in ix])
    y = torch.stack([data[i+1:i+max_len+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data, batch_size, max_len, device):
    out = {}
    model.eval()
    names = ['train','val']
    data = [train_data, val_data]
    for i in range(2):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data[i], batch_size, max_len, device)
            logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[names[i]] = losses.mean()
    model.train()
    return out






def _read_wiki(data_dir):
    """
    wiki.train.tokens is some text for which each line is a paragraph
    We read through wiki.train.tokens to generate a list of paragraphs, which are themselves
    a list of sentences (which is a list)
    We only keep the paragraphs with at least two sentences
    """
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs



