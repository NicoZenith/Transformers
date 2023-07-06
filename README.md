# Transformers

PyTorch re-implementation of transformers. 

### ``transformer_layers.py`
The main components of the transformer architecture is found in the `transformer_layers.py` file. The class `Head` defines the self-attention layer, with the possibility to adapt the attention mask to variable input sequences with padding (`valid_lens`), or stop the tokens from attending the future tokens (`forward_mask` variable). 


