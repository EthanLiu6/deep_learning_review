"""
single head attention
"""
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, ipt: Tensor, *args):  # please insure your input shape like:
        # [batch_size, seq_len, embedding_dim]
        super(SingleHeadAttention, self).__init__()
        self.embedding_dim = ipt.shape[-1]
        self.qkv = nn.Linear(self.embedding_dim, self.embedding_dim * 3)

    def forward(self, X):

        # 1. q,k,v   2.q @ k^t   3.mask(ignore)   4.softmax scale   5.attention res
        qkv: Tensor = self.qkv(X)
        query, key, value = torch.split(qkv, [self.embedding_dim, self.embedding_dim, self.embedding_dim])
        print(query.shape)
        print(key.shape)
        print(value.shape)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        scores = F.softmax(attention_scores, dim=-1)

        # if mask is not None:
        #    pass
        # if dropout is not None:
        #    pass

        attention_res = torch.matmul(attention_scores, value)

        return attention_res, attention_scores





