# The implementation of transformer encoder from scratch
# https://www.youtube.com/watch?v=g3sEsBGkLU0&list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4&index=6

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 512 # intermediate embedding dimensions
num_heads = 8 # multi-head attention
dropout_rate = 0.1 # randomly turn off 10% of the neurons. This forces the network to learn along different paths. (Regularization)
                    # Which help the model to better "generalize" data instead of accidentally memorize certain data.
batch_size = 30     # Pass multiple examples (sequences) at the same time. These multiple examples constitute a "batch".
                    # Faster training & More stable training
                    # Loss function be computed and gradients update ONLY after seeing 30 examples -> mini-batch gradient descent
max_sequence_length = 200 # Longest sentence can be passed in at a time through encoders.
                          # If a sentence length is < max_seq_len, we pad the sentence by 0. (e.g. My name is Goro <pad> <pad> <pad> <pad> given max_len=8)
ffn_hidden = 2048   # Feed Forward Network: Expanding the number of neurons from 512 to ffn to "learn additional information"
num_layers = 5      # Number of encoder units in the architecture (Nx)



# Scaled Dot-Product Attention
# Only transpose the last two dimensions - [B, S, E] to [B, E, S]
# E - Embeddings dimension (512)
# S - Sequence length (200)
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size()[-1] # query vector dimension - # B, num_heads, L, E (30, 8, 200, 64) - 64
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # (30, 8, 200, 200)
    if mask is not None:
        scaled = scaled + mask # mask is only used in decoder - avoid predictions from looking at the future
                               # mask - upper triangular to -inf and lower triangular to 0 
                               # Broadcasting ADD - Just need to match the last N dimension (200, 200)
    attention = F.softmax(scaled, dim=-1) 
    values = torch.matmul(attention, v) # values vectors know how much attention it should pay to all the other words in that sentence. More context awareness
                                        # (30, 8, 200, 200) X (30, 8, 200, 64) = (30, 8, 200, 64)
    return values, attention    # (30, 8, 200, 64), (30, 8, 200, 200) - attention between words in the sequences

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model        # 512
        self.num_heads = num_heads    # 8
        self.head_dim = d_model // num_heads # 64
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model) # feed-forward layer (input_dim, 3*512)
        self.linear_layer = nn.Linear(d_model, d_model)    # (512, 512)

    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size() # 30 x 200 x 512
        qkv = self.qkv_layer(x) # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3)   # 30, 8, 200, 192
        q, k, v, = qkv.chunk(3, dim=-1) # 30, 8, 200, 64
        values, attention = scaled_dot_product_attention(q, k, v, mask) # 30 x 8 x 200 x 64
        values = values.reshape(batch_size, sequence_length, num_heads * self.head_dim) # Concatenate values - 30, 200, 512
        out = self.linear_layer(values) # 30, 200, 512
        return out

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape # embedding dimension - 512
        self.eps = eps # Avoid division by zero error.
        self.gamma = nn.Parameter(torch.ones(self.parameters_shape)) # [512] learnable parameters
        self.beta = nn.Parameter(torch.zeros(self.parameters_shape)) # [512] learnable parameters
    
    def forward(self, inputs): # inputs shape: 30, 200, 512
        dims = [-(i+1) for i in range(len(self.parameters_shape))] # [-1] - on the 'layer' dimension
        mean = inputs.mean(dim=dims, keepdim=True)                 # 30, 200, 1 - one mean per word
        var = ((inputs-mean)**2).mean(dim=dims, keepdim=True)      # 30, 200, 1 - one std per word
        std = (var + self.eps).sqrt()                              # 30, 200, 1
        y = (inputs - mean) / std                                  # 30, 200, 512 (Broadcasting mean & std matrices)
        out = self.gamma * y + self.beta                           # 30, 200, 512
        return out

class PositionwiseFeedForward(nn.Module): # The FeedForward Layer after Multi-head Attention & Add & Norm
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) # Expand the dimension to learn additional information 512, 2048
        self.linear2 = nn.Linear(hidden, d_model) # 2048, 512
        self.relu = nn.ReLU()                     # 2048 - Activation functions help neural networks to learn more complex patterns by capturing non-linearities.
        self.dropout = nn.Dropout(p=drop_prob)    # 2048   ReLU - piecewise linear function (made of multiple straight lines)
                                                  #        Act as own regularizer - turns off some neurons (x<0) -> leakly relu helps vanishing gradient problem

    def forward(self, x):    # 30, 200, 512
        x = self.linear1(x)  # 30, 200, 2048
        x = self.relu(x)     # 30, 200, 2048
        x = self.dropout(x)  # 30, 200, 2048
        x = self.linear2(x)  # 30, 200, 512
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(input_dim=d_model, d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual_x = x                      # 30, 200, 512
        x = self.attention(x, mask=None)    # 30, 200, 512
        x = self.dropout1(x)                # 30, 200, 512
        x = self.norm1(x + residual_x)      # 30, 200, 512
        residual_x = x # update residual for next skip connection
        x = self.ffn(x)                     # 30, 200, 512
        x = self.dropout2(x)                # 30, 200, 512
        x = self.norm2(x + residual_x)      # 30, 200, 512
        return x                            # 30, 200, 512 (much more context-aware)
    
# nn.Module is the super class.
# nn.Module allows the model to perform many operations behind the scene required for learning.
# Provides a lot of bootstrap codes for conveniences. Checkpoints to save the trained model.
class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) 
                                      for _ in range(num_layers)])
    def forward(self, x):
        x = self.layers(x)
        return x  # really good at encapsulating contexts
    
if __name__ == "__main__":
    # Create an encoder object
    encoder = Encoder(d_model, ffn_hidden, num_heads, dropout_rate, num_layers)
    embedding_dim = d_model
    x = torch.randn(batch_size, max_sequence_length, embedding_dim)
    out = encoder(x)
    print(f"Encoder output shape: {out.shape}")
    # print(out)