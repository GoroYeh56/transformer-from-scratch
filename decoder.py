import torch
import torch.nn as nn

from encoder import LayerNormalization, MultiHeadAttention, PositionwiseFeedForward
from encoder import scaled_dot_product_attention as scaled_dot_product

d_model = 512 # embedding dimension. Each word token is 512x1 vector
num_heads = 8
drop_prob = 0.1 # Randomly turning off 10% of neurons. Helps the model learn from different paths during backprop -> help generalize the model
batch_size = 30 # Feed 30 input sequences at a time
max_sequence_length = 200 
ffn_hidden = 2048 # Output dimension of the FeedForward Layer
num_layers = 5 # Number of decoder blocks to be cascaded
               # Why do we need to cascade? To capture the intricacy of language
               # The more complext the patterns can be in language, the more intricate you want the architecture to be.

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2*d_model) # From encoder block
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        batch_size, max_seq_len, embedding_dim = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, max_seq_len, num_heads, 2*self.head_dim)
        q = q.reshape(batch_size, max_seq_len, num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3) # B, num_heads, L, 2* (E//num_heads)
        q = q.permute(0, 2, 1 ,3)   # B, num_heads, L, (E//num_heads)
        k, v = kv.chunk(2, dim=-1)  # B, num_heads, L, E//num_heads (30, 8, 200, 64)
        values, attention = scaled_dot_product(q, k, v, mask)    # values: (B, L, E) attention: (B, L, L)
        values = values.reshape(batch_size, max_seq_len, d_model) # Concatenate heads - (B, num_heads, L, E//num_heads) to (B, L, E)
        out = self.linear_layer(values) # B, L, E
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(input_dim = d_model, d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, decoder_mask):
        # x: encoder block output value vectors
        # y: decoder input (ground-truth for current input batch) (After positional encoding)
        _y = y              # For residual addition!
        y = self.self_attention(y, mask=decoder_mask)   # 30 x 200 x512
        y = self.dropout1(y)
        y = self.norm1(y + _y) # Residual connection
        y_ = y # For residual connection

        y = self.encoder_decoder_attention(x, y, mask=None) # No mask needed for cross-attention
        y = self.dropout2(y)        
        y = self.norm2(y + _y)
        y_= y           # 30 x 200 x 512

        y = self.ffn(y)             # 30 x 200 x 512
        y = self.dropout3(y)        # 30 x 200 x 512
        y = self.norm3(y + _y)      # 30 x 200 x 512
        return y

# The reason we use another SequentialDecoder class is because: nn.Sequential can only pass ONE parameter to forward(self, x)!
# Implement own wrapper - extends sequential by extracting x, y, mask from *inputs
class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask) # x & mask are the same. Only y is updated iafter each decoder block
        return y       

class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, mask):
        # x: 30 x 200 x 512
        # y: 30 x 200 x 512
        # mask: 200 x 200
        y = self.layers(x, y, mask)
        return y
    

if __name__ == "__main__":
    x = torch.randn(batch_size, max_sequence_length, d_model) # English sentence positional encoded. Output from Encoder block
    y = torch.randn(batch_size, max_sequence_length, d_model) # Mandarin sentence positional encoded
    mask = torch.full([max_sequence_length, max_sequence_length], float('-inf')) # Lookahead mask for decoder - during training, 
    mask = torch.triu(mask, diagonal=1)                                           # Not allowed to see future words - prevent 'cheating' from happening
    decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
    out = decoder(x, y, mask)
    print(f"Decoder output shape: {out.shape}")