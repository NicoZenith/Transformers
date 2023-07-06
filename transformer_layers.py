import torch 
import math
import torch.nn as nn
from torch.nn import functional as F


def create_mask(valid_lens, max_len):
    """
    Creates an attention mask that adapts to the actual length of the sequence, padded with 0 until max_len
    :valid_lens: stores the length of the input sequences, dimension (batch_size, max_len)
    :max_len: max length of the input sequence, padded with 0 if necessary
    :return: mask matrix for each data point of the batch,
    with a square of ones in the upper left part of size valid_lens[i]*valid_lens[i], dimension (batch_size, max_len, max_len)
    """
    batch_size = len(valid_lens)
    mask = torch.arange(max_len).expand(batch_size, max_len) < valid_lens.unsqueeze(1)
    mask = mask.unsqueeze(1).float()
    mask = mask * mask.transpose(1, 2)
    return mask


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, d_model, dropout):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        # we don't need to use an triangular inferior matrix as it is bidirectional.
        # However, we want to adapt the mask to the input sequence length
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid_lens=None, forward_mask=None):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        batch_size, max_len, d_model = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        self.tril = torch.tril(torch.ones(max_len, max_len))  # triangular inferior matrix for GPT 

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        if forward_mask is not None:
            wei = wei.masked_fill(self.tril == 0, float('-inf')) # (B, T, T)

        if valid_lens is not None:
            self.mask = create_mask(valid_lens, max_len)
            wei = wei.masked_fill(self.mask == 0, float('-inf')) # (B, T, T)
            row_mask = torch.isinf(wei).all(dim=2)
            # Apply softmax to the whole row while avoiding rows with only -inf 
            wei = torch.where(row_mask.unsqueeze(2),  torch.zeros_like(wei), torch.softmax(wei, dim=2))
        else:
            wei = torch.softmax(wei, dim=2)
            
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, d_model, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, d_model, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid_lens=None, forward_mask=None):
        out = torch.cat([h(x, valid_lens, forward_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, d_model, num_heads, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = d_model // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, d_model, dropout)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, valid_lens=None, forward_mask=None):
        x = x + self.sa(self.ln1(x), valid_lens, forward_mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function"
    def __init__(self, n_embd, p_drop, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=p_drop)

        pe = torch.zeros(max_len, n_embd, device=device)
        position = torch.arange(0, max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2, device=device) / n_embd * (-math.log(10000.0))  ) # T/2
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        output = x + nn.Parameter(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(output)


