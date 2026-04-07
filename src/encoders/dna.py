"""DNAEncoder — Section 3.1. Transformer + attentive pooling over k-mer tokens."""

import torch
import torch.nn as nn


class AttentivePooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        weights = torch.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)


class DNAEncoder(nn.Module):
    def __init__(self, vocab_size: int = 4**4 + 1, embed_dim: int = 256,
                 n_heads: int = 8, n_layers: int = 4, output_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = AttentivePooling(embed_dim)
        self.proj = nn.Linear(embed_dim, output_dim)

    def forward(self, kmer_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(kmer_ids)
        x = self.transformer(x)
        x = self.pool(x)
        return self.proj(x)
