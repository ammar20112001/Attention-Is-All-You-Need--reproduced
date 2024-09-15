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