import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    # 1. q * k_T / sqrt(d_k),  2. mask, 3.softmax, 4.scores * v
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    scores = torch.softmax(attention_scores, dim=-1)

    attention = torch.matmul(scores, value)
    return attention, scores


if __name__ == '__main__':
    batch_size = 2
    seq_len = 5
    d_model = 4

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    # 创建 mask（1 表示有效，0 表示 padding）
    mask = torch.tril(torch.ones(batch_size, seq_len, seq_len))
    mask1 = torch.tensor([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]
    ])
    print(mask==mask1)

    # 计算注意力
    output, scores = scaled_dot_product_attention(query, key, value, mask)
    print("Attention shape:", output.shape)
    print("Attention scores:\n", scores)
