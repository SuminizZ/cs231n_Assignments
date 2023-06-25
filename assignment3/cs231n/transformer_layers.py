import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import time 

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        pe = torch.zeros(1, max_len, embed_dim)

        # with a for-loop
        # for i in range(max_len):
        #     for j in range(embed_dim//2):
        #         pe[0, i, 2*j] = np.sin(i*(max_len**(-2*j/embed_dim)))
        #         pe[0, i, 2*j+1] = np.cos(i*(max_len**(-2*j/embed_dim)))
        # toc = time.time()

        # w/o explicit for-loop : faster
        p_seq = torch.arange(0, max_len).unsqueeze(1)
        p_idx = 10000**(torch.arange(0, embed_dim//2)*(-2/embed_dim))
        outer = p_seq*p_idx

        even_idx = torch.arange(0, embed_dim//2)*2
        odd_idx = torch.arange(0, embed_dim//2)*2 + 1

        pe[:, :, even_idx] = torch.sin(outer)
        pe[:, :, odd_idx] = torch.cos(outer)
        toc = time.time()

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        output = self.dropout(x + self.pe[0, :S, :])

        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as sthe value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape   # S = T in self-attention
        output = torch.empty((N, S, E))
        D = E//self.n_head
        
        query = self.query(query).reshape((N, S, self.n_head, D)).permute(0, 2, 1, 3)
        key = self.key(key).reshape((N, T, self.n_head, D)).permute(0, 2, 3, 1)
        value = self.value(value).reshape((N, T, self.n_head, D)).permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)/math.sqrt(D)   # N, H, S, T

        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (S-T,T-S), 'constant', 0)
            attention = attention.masked_fill(attn_mask == 0, -1e8)    # softmax 

        attention_prob = self.attn_drop(F.softmax(attention, dim=-1))  

        output = torch.matmul(attention_prob, value)
        output = output.transpose(1,2).contiguous().view(N, S, E)
        output = self.proj(output)
        
        return output


