import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Step 1: Linear projections and reshape
        query = self.query(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,
                                                                                           2)  # (batch, heads, seq_len, d_k)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Step 2: Scaled dot-product attention
        energy = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, heads, seq_len, seq_len)

        # 确保 mask 形状正确 (batch_size, 1, seq_len, seq_len)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            energy = energy.masked_fill(mask == 0, float('-1e10'))

        attention = torch.nn.functional.softmax(energy, dim=-1)

        # Step 3: Weighted sum of values
        out = torch.matmul(attention, value)  # (batch, heads, seq_len, d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Step 4: Final linear projection
        out = self.fc_out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self attention with residual connection
        attention_out = self.self_attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout1(attention_out))

        # Feedforward network with residual connection
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + self.dropout2(ffn_out))

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # Self attention
        self_attention_out = self.self_attention(x, x, x, tgt_mask)
        x = self.layer_norm1(x + self.dropout1(self_attention_out))

        # Cross attention (from encoder output)
        cross_attention_out = self.cross_attention(x, enc_out, enc_out, src_mask)
        x = self.layer_norm2(x + self.dropout2(cross_attention_out))

        # Feedforward network
        ffn_out = self.ffn(x)
        x = self.layer_norm3(x + self.dropout3(ffn_out))

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, num_heads, num_encoder_layers, d_ff, dropout)
        self.decoder = Decoder(vocab_size, d_model, num_heads, num_decoder_layers, d_ff, dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return out


if __name__ == '__main__':
    vocab_size = 1000
    d_model = 512
    seq_len = 10
    batch_size = 2

    embedding_layer = EmbeddingLayer(vocab_size, d_model)
    pos_encoding = PositionalEncoding(d_model)

    sample_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    embed_out = embedding_layer(sample_input)
    pos_out = pos_encoding(embed_out)

    print("Embedding output shape:", embed_out.shape)  # Expected: (batch_size, seq_len, d_model)
    print("Positional encoding output shape:", pos_out.shape)  # Expected: (batch_size, seq_len, d_model)
    "---------------------------------"

    num_heads = 8
    multi_head_attention = MultiHeadAttention(d_model, num_heads)

    sample_input = torch.rand(batch_size, seq_len, d_model)
    attn_out = multi_head_attention(sample_input, sample_input, sample_input)

    print("Multi-head attention output shape:", attn_out.shape)  # Expected: (batch_size, seq_len, d_model)
    "-------------------------"

    d_ff = 2048
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

    mask = torch.ones(batch_size, seq_len, seq_len)  # 这里可以使用全1的掩码
    encoder_out = encoder_layer(sample_input, mask)

    print("Encoder layer output shape:", encoder_out.shape)  # Expected: (batch_size, seq_len, d_model)
    "-------------------------"

    decoder_layer = DecoderLayer(d_model, num_heads, d_ff)

    encoder_out = torch.rand(batch_size, seq_len, d_model)
    tgt_mask = torch.ones(batch_size, seq_len, seq_len)
    decoder_out = decoder_layer(sample_input, encoder_out, mask, tgt_mask)

    print("Decoder layer output shape:", decoder_out.shape)  # Expected: (batch_size, seq_len, d_model)
    "-------------------------"

    num_encoder_layers = 6
    num_decoder_layers = 6

    transformer = Transformer(vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff)

    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))

    src_mask = torch.ones(batch_size, seq_len, seq_len)
    tgt_mask = torch.ones(batch_size, seq_len, seq_len)

    output = transformer(src, tgt, src_mask, tgt_mask)

    print("Transformer output shape:", output.shape)  # Expected: (batch_size, seq_len, vocab_size)
