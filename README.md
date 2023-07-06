# Transformers

PyTorch re-implementation of transformers. 

### Constructing transformer architecture
The `transformer_layers.py` file contains the main components of the transformer architecture.
The class `Head` defines the self-attention layer, with the possibility to adapt the attention mask to variable input sequences with padding (`valid_lens`), or stop the tokens from attending future tokens (`forward_mask` variable). 
The class `MultiHeadAttention` creates multiple attention heads in parallel. 
The class `FeedForward` defines a linear layer followed by non-linearity applied to each encoded token (through the self-attention layer). 
The class `Block` defines a whole transformer block, containing a multi-head attention, a feedforward layer, and layer-normalization layer. 
Finally, a class `PositionalEncoding` creates the positional encoding necessary to provide sequence information concatenated to the input to the transformer.

### Training transformers on text-generation: GPT
The `GPT_text_generation.ipynb` notebook tests the transformer architecture on a GPT-like task, consisting of predicting the next token given a sequence of tokens. After training, the GPT network should be able to generate Shakespeare-like text. 

### Pretraining transformers: BERT 
The `BERT_Pretraining.ipynb` notebook adapts the transformer architecture for the BERT pre-training tasks, i.e., Mask Language Modeling (MLM) and Next Sentence Prediction. It also processes the WikiText2 dataset to obtain input sequences fitting the pretraining tasks, e.g., providing two consecutive sentences or not for NSP, randomly mask some tokens and provide their actual value as a target (MLM). 

### Fine-tuning BERT to sentiment analysis
The `BERT_FineTuning_Tweet_Classification.ipynb` uses a pre-trained BERT from Hugging-Face and fine-tunes it for a binary classification task of sentiment analysis, consisting of deciphering whether a tweet contains a disaster-like message or not. 









