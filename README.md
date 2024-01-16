# Transformers

PyTorch re-implementation of transformers. 

### Constructing transformer architecture
The `transformer_layers.py` file contains the main components of the transformer architecture.
The class `Self_Attention` defines the self-attention layer, with the possibility to adapt the attention mask to variable input sequences with padding (`valid_lens`), or stop the tokens from attending future tokens (`forward_mask` variable). 
The class `MultiHeadAttention` creates multiple attention heads in parallel. 
The class `FeedForward` defines a linear layer followed by non-linearity applied to each encoded token (through the self-attention layer). 
The class `Block` defines a whole transformer block, containing a multi-head attention, a feedforward layer, and layer-normalization layer. 
Finally, a class `PositionalEncoding` creates the positional encoding necessary to provide sequence information concatenated to the input to the transformer.

### Decoder only for next-word prediction: GPT
The `GPT_text_generation.ipynb` notebook tests the transformer architecture on a GPT-like task, consisting of predicting the next token given a sequence of tokens. After training, the GPT network should be able to generate Shakespeare-like text. A python script version is provided in `train_GPT.py`.

### Encoder only pretraining: BERT 
The `BERT_Pretraining.ipynb` notebook adapts the transformer architecture for the BERT pre-training tasks, i.e., Mask Language Modeling (MLM) and Next Sentence Prediction. It also processes the WikiText2 dataset to obtain input sequences fitting the pretraining tasks, e.g., providing two consecutive sentences or not for NSP, randomly mask some tokens and provide their actual value as a target (MLM). 

### Encoder-Decoder transformers: Seq2Seq 
The `seq2seq.ipynb` notebook uses the transformer architecture for sequence to sequence learning (e.g., translation). Here, a simple sequence to reversed-sequence is implemented to test whether the model works. A python script version is provided in `train_seq2seq_reverse.py`. I also test the model on a more complex task, i.e., french-to-english translation. The code is provided in `train_seq2seq_translation.py`. 
To test the trained model, use `generate_seq2seq_reverse.py` for sequence reversal and `generate_seq2seq_translation.py`. 
To adapt the transformer architecture for this task, we provide additional classes for cross attention in the file `transformer_layers.py`:
The class `Cross_Attention` defines the cross-attention layer between the encoder output and the decoder embedded tokens. It adapts the cross attention mask to the variable input and target sequences (`valid_lens_x` and `valid_lens_y`). 
The class `MultiHeadCrossAttention` uses the cross attention with multiple heads. 
The class `Decoder_Block` stacks self-attention, cross-attention and feedforward layers into a decoder-block. 
The class `Seq2Seq` constructs the encoder decoder transformer with cross attention layers. 

### Prediction of embeddings: GPT2 and trained Seq2Seq
I also provide the code to predict the embedding on a pre-trained GPT2 from Huggingface (`gpt2_predictions.py`). The embeddings from our own train transformer model are provided by the file `algo3_predictions.py`, using hooks to save activations. 


### Fine-tuning BERT to sentiment analysis
The `BERT_FineTuning_Tweet_Classification.ipynb` uses a pre-trained BERT from Hugging-Face and fine-tunes it for a binary classification task of sentiment analysis, consisting of deciphering whether a tweet contains a disaster-like message or not. 









