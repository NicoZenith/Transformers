# Transformers

PyTorch re-implementation of transformers. 

#### `transformer_layers.py`
The main components of the transformer architecture is found in the `transformer_layers.py` file. 

The class `Head` defines the self-attention layer, with the possibility to adapt the attention mask to variable input sequences with padding (`valid_lens`), or stop the tokens from attending future tokens (`forward_mask` variable). 

The class `MultiHeadAttention` creates multiple attention heads in parallel. 

The class `FeedForward` defines a linear layer followed by non-linearity applied to each encoded token (through the self-attention layer). 

The class `Block` defines a whole transformer block, containing a multi-head attention, a feedforward layer, and layer-normalization layer. 

Finally, a class `PositionalEncoding` creates the positional encoding necessary to provide sequence information concatenated to the input to the transformer.







