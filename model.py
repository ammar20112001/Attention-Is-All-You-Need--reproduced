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