
import torch 
import os
import random
import numpy as np 
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from torchtext.data.utils import get_tokenizer



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



