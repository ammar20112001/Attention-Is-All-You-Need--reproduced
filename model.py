# The code is referred from the Github Repo:
# https://github.com/hkproj/pytorch-transformer/blob/main/model.py

import torch # type: ignore
import torch.nn as nn # type: ignore
import math

class Embeddings(nn.Module):
    '''
    PyTorch Documentation: 
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    '''

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)

    def forward(self, x):
        '''
        Arguments:
            x -->   (B, S)                B: Batch Size; S: Sequence Length

        Return:
            e -->   (B, S, d_model)       d_model: Model's Dimension
        '''
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncodings(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a 2 dimensional array of Positional Encodings
        # pe --> (S, d_model)
        pe = [[math.sin(p/(10_000**((2*i)/self.d_model))) if i % 2 == 0 else math.cos(p/(10_000**((2*i)/self.d_model))) for i in range(self.d_model)] for p in range (self.seq_len)]
        
        # Convert pe list to Torch Tensor
        pes = torch.FloatTensor(pes)
        # Add a batch dimension to the positional encoding
        # pe --> (1, S, d_model)
        pe = pe.unsqueeze(0)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        Arguments:
            x -->   (B, S, d_model)

        Return:
            PE -->  (B, S, d_model)
        '''
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # This operation automatically broadcasts the Positional Encodings (pe) across all batches of x
        return self.dropout(x)

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, h, mask, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.mask = mask
        self.dropout = dropout

        self.dk = d_model // h
        self.dv = d_model // h
        
        # Create weight matrices for Quer, Key, and Value
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v):
        '''
        Arguments:
            q --> (B, S, d_model)
            k --> (B, S, d_model)
            v --> (B, S, d_model)

        Return:
            x --> (B, S, d_model)
            a --> 
        
        Operations:
            - Query, key, and value are multiplyed with matrices w_q, w_k, and w_v respectively
                q/k/v * w_q --> (B, S, d_model) * (d_model, d_model) --> (B, S, d_model)
            
            - Query, key, and value are divided into `h` number of HEADS
                q/k --> (B, S, d_model) --> (B, S, h, dk) --> (B, h, S, dk)
                v --> (B, S, d_model) --> (B, S, h, dv) --> (B, h, S, dv)
            
            - Attention scores are calculated
                HEAD(s) --> softmax( (q * k.T) / sqrt(dk)) * v 
                        --> ((B, h, S, dk) * (B, h, dk, S)) * (B, h, S, dv)
                        --> (B, h, S, S) * (B, h, S, dv)
                        --> (B, h, S, dv)

            - All HEADs are concatenated 
                MultiHead   --> Concat(HEAD_1, ..., HEAD_h) 
                            --> (S, dk*h) 
                            --> (S, d_model)

            - MultiHead * w_o   --> (S, d_model) 
                                --> (d_model, d_model) 
                                --> (S, d_model)   
        '''
        
        # Query, key, and value are multiplyed with matrices w_q, w_k, and w_v respectively
        # query/key/value * w_q --> (B, S, d_model) * (d_model, d_model) --> (B, S, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Query, key, and value are divided into `h` number of HEADS
        # q/k --> (B, S, d_model) --> (B, S, h, dk) --> (B, h, S, dk)
        # v --> (B, S, d_model) --> (B, S, h, dv) --> (B, h, S, dv)
        query = query.view(query[0], query[1], self.h, self.dk).transpose(1, 2)
        key = key.view(key[0], key[1], self.h, self.dk).transpose(1, 2)
        value = value.view(value[0], value[1], self.h, self.dv).transpose(1, 2)

        # Attention scores are calculated
        # HEAD(s) --> softmax( (q * k.T) / sqrt(dk)) * v 

        # --> ((B, h, S, dk) * (B, h, dk, S)) --> (B, h, S, S)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.dk)

        if self.mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(self.mask == 0, -1e9)
        
        # Apply softmax --> (B, h, S, S)
        attention_scores = attention_scores.softmax(dim=-1) 

        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)

        # (B, h, S, S) * (B, h, S, dv) --> (B, h, S, dv)
        attention = attention_scores @ value

        # All HEADs are concatenated
        # Concat(HEAD_1, ..., HEAD_h)
        # (B, h, S, dv) --> (B, S, h, dv) --> (batch, S, d_model)
        x = attention_scores.transpose(1, 2).contiguous().view(attention_scores[0], -1, self.h*self.dv)

        # Multiply by w_o
        # (B, S, d_model) * (d_model, d_model) --> (B, S, d_model)
        return self.w_o(x)

class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_fc: int, dropout: float) -> None:
        super().__init__()
        # Initialize linear layers
        self.fc1 = nn.Linear(d_model, d_fc)
        self.fc2 = nn.Linear(d_fc, d_model)
        self.dropout = dropout

    def forward(self, x):
        # (B, S, d_model) --> (B, S, d_ff) --> (B, S, d_model)
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class LayerNormalization(nn.Module):
    
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__(self)
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # Learnable parameter
        self.beta = nn.Parameter(torch.zeros(d_model)) # Learnable parameter
        #self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Calcualte mean of x
        # x --> (B, S, d_model)
        mean = x.mean(dim=-1, keepdim=True) # x --> (B, S, 1)

        # Calculate standard deviation
        std = x.std(dim=-1, keepdim=True) # x --> (B, S, 1)

        # Return normalized values
        # (B, S, d_model)
        return self.alpha((x - mean) / (math.sqrt(std**2 + self.eps))) + self.beta
    
class ResidualConnection(nn.Module):

    def __init__(self, d_model: int, dropout: float):
        super().__init__(self)
        self.norm = LayerNormalization(d_model)
        self.dropout(dropout)

    def forward(self, x, sublayer):
        x = x + self.dropout((sublayer(self.norm(x))))

class EncoderBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 multi_head_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForward,
                 dropout: float
                 ):
        super().__init__(self)
        self.multi_head_attention_block = multi_head_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = dropout
        self.residual_connection_block = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection_block[0](x, lambda x: self.multi_head_attention_block(x, x, x, src_mask))
        x = self.residual_connection_block[1](x, self.feed_forward_block)
        return x

class DecoderBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 multi_head_attention_block: MultiHeadAttention,
                 cross_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForward,
                 dropout: float
                 ):
        super().__init__(self)
        self.multi_head_attention_block = multi_head_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = dropout
        self.residual_connection_block = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection_block[0](x, lambda x: self.multi_head_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection_block[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection_block[2](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, d_model, layers: nn.ModuleList) -> None:
        super().__init__(self)
        self.layers = layers
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):

    def __init__(self, d_model, layers: nn.ModuleList) -> None:
        super().__init__(self)
        self.layers = layers
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
