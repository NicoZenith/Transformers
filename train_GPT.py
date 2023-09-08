import torch
import torch.nn as nn
import numpy as np
from transformer_layers import *
from utils import * 
from torch.nn import functional as F
from torch.utils.data import DataLoader



batch_size = 64
max_len = 16
num_layers = 6
num_heads = 6
dropout = 0.2
d_model = 384
num_epochs = 8
learning_rate = 1e-5
show_every = 200



with open('beatles.txt', 'r') as file:
    text = file.read()

print("length of dataset in characters", len(text))


# Tokenize the text using the custom tokenizer
word_tokens = custom_tokenizer(text)

# Collect unique tokens from the dataset
vocab = set()
for data_point in word_tokens:
    vocab.add(data_point)

vocab = list(vocab)
vocab = ['<pad>'] + vocab
vocab_size = len(vocab)
print("Size of vocabulary =", vocab_size)
vocab_dict = {token: index for index, token in enumerate(vocab)}
vocab_dict_reverse = {index: token for index, token in enumerate(vocab)}

# Encoding and decoding functions
encode = lambda sentence: [vocab_dict[word] for word in custom_tokenizer(sentence)]

# Modify the decode function to return the text with line breaks
def decode(l):
    decoded_sentence = ""
    for integer in l:
        token = vocab_dict_reverse[integer]
        if token == '<br>':
            decoded_sentence += '\n'  # Add line break
        else:
            decoded_sentence += token + ' '
    return decoded_sentence.strip()  # Remove trailing space


# Test the encoding and decoding
encoded_text = encode(text)
decoded_text = decode(encoded_text[:100])


n = int(0.9*len(encoded_text)) # first 90% will be train, rest val
train_data = encoded_text[:n]
val_data = encoded_text[n:]



train_dataset = GPT_Dataset(train_data, max_len)
val_dataset = GPT_Dataset(val_data, max_len)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT(d_model, dropout, max_len, vocab_size, num_heads, num_layers, device)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

train_losses = []
val_losses = []


for e in range(num_epochs):
    store_train_losses = []
    store_val_losses = []
    for i, batch in enumerate(train_dataloader):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        model.zero_grad()

        logits = model(inputs)
        loss = model.compute_loss(logits, targets)
        loss.backward()
        optimizer.step()

        store_train_losses.append(loss.item())

    for i, batch in enumerate(val_dataloader):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        model.zero_grad()
        with torch.no_grad():
            logits = model(inputs)
            loss = model.compute_loss(logits, targets)

        store_val_losses.append(loss.item())

    train_losses.append(np.mean(store_train_losses))
    val_losses.append(np.mean(store_val_losses))

    print("Epoch {}: Train Loss: {:.4f}, Validation Loss: {:.4f}".format(e, train_losses[-1], val_losses[-1]))


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=300, valid_lens=None, forward_mask=True)[0].tolist()))




