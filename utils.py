
import torch 
import os
import random



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
    for data in [train_data, val_data]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, max_len, device)
            logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[data] = losses.mean()
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