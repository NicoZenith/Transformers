import torch 
import math
import torch.nn as nn
from torch.nn import functional as F






class GPT(nn.Module):
    def __init__(self, d_model, dropout, max_len, vocab_size, num_heads, num_layers, device):
        super().__init__()
        self.device = device
        self.max_len = max_len
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(max_len, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.blocks = nn.Sequential(*[Block(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def compute_loss(self, logits, targets):
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return loss


    def forward(self, idx, forward_mask=True):
        """
        Takes a context sequence of tokens as input, and generates the next token. Can be used only for prediction (targets = None)
        or for training.
        :idx: context sequence of tokens
        :valid_lens: actual length of the sequence
        :forward_mask: for unidirectional attention
        :targets: batch of targets for the next token
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, d_model)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb # (B, T, d_model)
        for blk in self.blocks:
            x = blk(x, valid_lens=None, forward_mask=forward_mask)
        logits = self.lm_head(x)

        return logits



    def generate(self, idx, max_new_tokens, valid_lens=None, forward_mask=True):
        """
        Generate text by taking a context of size max_len and predicting the next token
        by sampling from the probability distribution obtained with the softmax on the whole vocabulary
        :idx: (B, T) array of indices in the current context
        :max_new_tokens: the number of new tokens we want to generate
        :return: generated sequence including the initial context
        """

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            # get the predictions
            logits = self(idx_cond, forward_mask)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim =1) # (B, T+1)
        return idx







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




class Self_Attention(nn.Module):
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
        self.tril = torch.tril(torch.ones(max_len, max_len)).to(x.device)  # triangular inferior matrix for GPT 

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        if forward_mask is not None:
            wei = wei.masked_fill(self.tril == 0, float('-inf')) # (B, T, T)

        if valid_lens is not None:
            self.mask = create_mask(valid_lens, max_len).to(x.device)
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
        self.heads = nn.ModuleList([Self_Attention(head_size, d_model, dropout) for _ in range(num_heads)])
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
        x = self.ln1(x + self.sa(x, valid_lens, forward_mask))
        x = self.ln2(x + self.ffwd(x))
        return x








class Seq2Seq(nn.Module):
    def __init__(self, d_model, dropout, max_len, vocab_size_source, vocab_size_target, num_heads, num_layers, device):
        super().__init__()
        self.device = device
        self.max_len = max_len
        self.num_layers = num_layers
        self.token_embedding_table_encoder = nn.Embedding(vocab_size_source, d_model)
        self.token_embedding_table_decoder = nn.Embedding(vocab_size_target, d_model)
        self.position_embedding = PositionalEncoding(d_model, dropout, max_len, device)
        self.lm_head = nn.Linear(d_model, vocab_size_target)

        self.encoder_blocks = nn.Sequential(*[Block(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.decoder_blocks = nn.Sequential(*[Decoder_Block(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def compute_loss(self, logits, labels, valid_lens_y):
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        labels = labels.view(B*T)
        # Create a mask based on valid_lens_y for each batch
        mask = torch.arange(T).expand(B, T) < valid_lens_y.view(-1, 1)
    
        # Flatten the mask
        mask = mask.view(-1)

        # Apply the mask to logits and targets
        logits_masked = logits[mask]
        labels_masked = labels[mask]
        loss = F.cross_entropy(logits_masked, labels_masked)
        return loss



    def forward(self, x, y, valid_lens_x, valid_lens_y):
        """
        Takes a context sequence of tokens as input, and generates the next token. Can be used only for prediction (targets = None)
        or for training.
        :idx: context sequence of tokens
        :valid_lens: actual length of the sequence
        :forward_mask: for unidirectional attention
        :targets: batch of targets for the next token
        """
        B, T = x.shape
        tok_emb_x = self.token_embedding_table_encoder(x) # (B, T, d_model)
        tok_emb_y = self.token_embedding_table_decoder(y) # (B, T, d_model)
        x = self.position_embedding(tok_emb_x)
        y = self.position_embedding(tok_emb_y)
        for i in range(self.num_layers):
            enc_blk = self.encoder_blocks[i]
            dec_blk = self.decoder_blocks[i]
            x = enc_blk(x, valid_lens=valid_lens_x, forward_mask=None)
            y = dec_blk(x, y, valid_lens_x, valid_lens_y, forward_mask=True)

        logits = self.lm_head(y)

        return logits

    
    def generate(self, x, valid_lens_x, max_new_tokens, eos, start):
        """
        Generate text by taking a context of size max_len and predicting the next token
        by sampling from the probability distribution obtained with the softmax on the whole vocabulary
        :idx: (B, T) array of indices in the current context
        :max_new_tokens: the number of new tokens we want to generate
        :return: generated sequence including the initial context
        """
        
        B, T = x.shape
        

        tok_emb_x = self.token_embedding_table_encoder(x) # (B, T, d_model)
        x = self.position_embedding(tok_emb_x)
        store_context = []
        for i in range(self.num_layers):
            enc_blk = self.encoder_blocks[i]
            x = enc_blk(x, valid_lens=valid_lens_x, forward_mask=None)
            store_context.append(x)
        
        y = torch.ones((B, 1), dtype=torch.long, device=self.device)*start
        for t in range(max_new_tokens-1):
            valid_lens_y = torch.ones(B, dtype=torch.long)*(t+1)
            y_emb = self.token_embedding_table_decoder(y) 
            y_emb =  self.position_embedding(y_emb)
            # get the predictions
            for i in range(self.num_layers):
                dec_blk = self.decoder_blocks[i]
                y_emb = dec_blk(store_context[i], y_emb, valid_lens_x, valid_lens_y, forward_mask=True)

            
            logits = self.lm_head(y_emb)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            y_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            #y_next = torch.argmax(logits, dim=-1)
            y = torch.cat((y, y_next), dim =1).long() # (B, T+1)
        
        truncated_sequences = []
        for i in range(B):
            sequence = y[i].tolist()
            for j in range(len(sequence)):
                if sequence[j] == eos:
                    break
            else:
                j +=1  
            sequence = sequence[:j]
            truncated_sequences.append(sequence)
                
        return truncated_sequences




def create_cross_attention_mask(input_valid_lens, target_valid_lens):
    """
    Creates a cross-attention mask that adapts to the actual lengths of input and target sequences.
    
    :param input_valid_lens: Valid lengths of the input sequences, dimension (batch_size,)
    :param target_valid_lens: Valid lengths of the target sequences, dimension (batch_size,)
    
    :return: Mask matrix for each data point of the batch,
             with a square of ones in the upper left part of size target_valid_lens[i] x input_valid_lens[i],
             dimension (batch_size, max_target_len, max_input_len)
    """
    batch_size = len(input_valid_lens)
    max_input_len = torch.max(input_valid_lens).item()
    max_target_len = torch.max(target_valid_lens).item()

    # Create masks directly based on the valid lengths using broadcasting
    input_indices = torch.arange(max_input_len).expand(batch_size, max_input_len)  # (B, T_input)
    target_indices = torch.arange(max_target_len).expand(batch_size, max_target_len)  # (B, T_target)

    input_mask = input_indices < input_valid_lens.unsqueeze(1)
    target_mask = target_indices < target_valid_lens.unsqueeze(1)
    
    mask = target_mask.unsqueeze(2) * input_mask.unsqueeze(1)  # Corrected order here
    mask = mask.float()
    
    return mask




class Cross_Attention(nn.Module):

    def __init__(self, head_size, d_model, dropout):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        # we don't need to use an triangular inferior matrix as it is bidirectional.
        # However, we want to adapt the mask to the input sequence length
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x, y, valid_lens_x, valid_lens_y, forward_mask=None):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        k = self.key(x)   # (B,T,hs)
        q = self.query(y) # (B,T,hs)
        self.tril = torch.tril(torch.ones(q.size(1), k.size(1))).to(x.device)  # triangular inferior matrix for GPT 

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        if forward_mask is not None:
            wei = wei.masked_fill(self.tril == 0, float('-inf')) # (B, T, T)


        self.mask = create_cross_attention_mask(valid_lens_x, valid_lens_y).to(x.device)[:, :q.size(1), :]


        wei = wei.masked_fill(self.mask == 0, float('-inf')) # (B, T, T)
        row_mask = torch.isinf(wei).all(dim=2)
        # Apply softmax to the whole row while avoiding rows with only -inf 
        wei = torch.where(row_mask.unsqueeze(2),  torch.zeros_like(wei), torch.softmax(wei, dim=2))
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out



class MultiHeadCrossAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, d_model, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Cross_Attention(head_size, d_model, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, valid_lens_x, valid_lens_y, forward_mask=None):
        out = torch.cat([h(x, y, valid_lens_x, valid_lens_y, forward_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out








class Decoder_Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, d_model, num_heads, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = d_model // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, d_model, dropout)
        self.ca = MultiHeadCrossAttention(num_heads, head_size, d_model, dropout)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, y, valid_lens_x, valid_lens_y, forward_mask=None):
        y = self.ln1(y + self.sa(y, valid_lens_y, forward_mask))
        context = self.ca(x, y, valid_lens_x, valid_lens_y, forward_mask)
        y = self.ln2(y + context)
        x = self.ln3(y + self.ffwd(y))
        return y


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





# # Example usage:
# batch_size = 3
# input_max_len = 10
# target_max_len = 12
# input_valid_lens = torch.tensor([7, 8, 6])
# target_valid_lens = torch.tensor([9, 10, 11])

# mask = create_cross_attention_mask(input_valid_lens, target_valid_lens, input_max_len, target_max_len)
# print(mask)