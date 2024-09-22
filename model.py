# The code is referred from the Github Repo:
# https://github.com/hkproj/pytorch-transformer/blob/main/model.py

import torch
import torch.nn as nn
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
        return self.embeddings(x) * math.sqrt(self.d_model)

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
        pe = torch.FloatTensor(pe)
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

    def __init__(self, d_model, h, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        self.dk = d_model // h
        self.dv = d_model // h
        
        # Create weight matrices for Quer, Key, and Value
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask):
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
        query = query.view(query.shape[0], query.shape[1], self.h, self.dk).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.dk).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.dv).transpose(1, 2)

        # Attention scores are calculated
        # HEAD(s) --> softmax( (q * k.T) / sqrt(dk)) * v 

        # --> ((B, h, S, dk) * (B, h, dk, S)) --> (B, h, S, S)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.dk)

        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax --> (B, h, S, S)
        attention_scores = attention_scores.softmax(dim=-1) 

        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)

        # (B, h, S, S) * (B, h, S, dv) --> (B, h, S, dv)
        attention = attention_scores @ value

        # All HEADs are concatenated
        # Concat(HEAD_1, ..., HEAD_h)
        # (B, h, S, dv) --> (B, S, h, dv) --> (batch, S, d_model)
        x = attention.transpose(1, 2).contiguous().view(attention.shape[0], -1, self.h*self.dv)

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
        super().__init__()
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
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    
class ResidualConnection(nn.Module):

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        print(x.shape)
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 multi_head_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForward,
                 dropout: float
                 ):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):

    def __init__(self, d_model, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        pass

    def forward(self, x):
        '''
        Arguument:
            x --> (B, S, d_model)

        Return:
            x --> (B, S, V)
        '''
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self,
                 src_embedding_layer: Embeddings,
                 tgt_embedding_layer: Embeddings,
                 pes_layer: PositionalEncodings,
                 encoder: Encoder,
                 decoder: Decoder,
                 projection_layer: ProjectionLayer):
        super().__init__()
        # Copy layer classes
        self.src_embedding_layer = src_embedding_layer
        self.tgt_embedding_layer = tgt_embedding_layer
        self.pes_layer = pes_layer
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encode(self, x: torch.Tensor, src_mask: torch.Tensor):
        x = self.src_embedding_layer(x)
        x = self.pes_layer(x)
        return self.encoder(x, src_mask) # (B, S, d_model)
    
    def decode(self, y: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        y = self.tgt_embedding_layer(y)
        y = self.pes_layer(y)
        return self.decoder(y, encoder_output, src_mask, tgt_mask) # (B, S, d_model)
    
    def project(self, x: torch.Tensor):
        return self.projection_layer(x) # (B, S, V)
    
def build_transformer(
        d_model: int = 512,
        heads: int = 8,
        n_stack: int = 6,
        max_seq_len: int = 512,
        src_vocab_size: int = 40_000,
        tgt_vocab_size: int = 40_000,
        dropout: float = 0.1,
        d_fc: int = 2048
        ):
    
    # Create Embedding layer instances
    src_embedding_layer = Embeddings(d_model, src_vocab_size)
    tgt_embedding_layer = Embeddings(d_model, tgt_vocab_size)

    # Create Positional Encoding layer instance
    pes_layer = PositionalEncodings(d_model, max_seq_len, dropout)

    encoder_blocks = [] # n-stack of Encoder blocks
    decoder_blocks = [] # n-stack of Decoder blocks

    for _ in range(n_stack):
        multi_head_attention_block = MultiHeadAttention(d_model, heads, dropout)
        feed_forward_block = FeedForward(d_model, d_fc, dropout)
        encoder_blocks.append(
            EncoderBlock(d_model, multi_head_attention_block, feed_forward_block, dropout)
            )

    for _ in range(n_stack):
        multi_head_attention_block = MultiHeadAttention(d_model, heads, dropout)
        cross_attention_block = MultiHeadAttention(d_model, heads, dropout)
        feed_forward_block = FeedForward(d_model, d_fc, dropout)
        decoder_blocks.append(
            DecoderBlock(d_model, multi_head_attention_block, cross_attention_block, feed_forward_block, dropout)
            )
        

    # Create complete Encoder and Decoder layer instances
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create Projection layer instance
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create Transformer instance
    transformer = Transformer(
        src_embedding_layer,
        tgt_embedding_layer,
        pes_layer,
        encoder,
        decoder,
        projection_layer
        )
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

if __name__ == '__main__':
    transformer = build_transformer()
    print(transformer, 
          f'Number of parameters: {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}', 
          sep='\n\n')