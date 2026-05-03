"""DNA encoder: k-mer Transformer + attentive pooling (Sect. 3.1)."""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


NUCLEOTIDES = ["A", "C", "G", "T", "N"]


def kmer_vocab(k: int) -> dict:
    if k < 1 or k > 8:
        raise ValueError(f"k must be in [1, 8], got {k}")
    vocab = {"<PAD>": 0, "<UNK>": 1}
    bases = ["A", "C", "G", "T"]

    def _gen(prefix: str, depth: int) -> Iterable[str]:
        if depth == 0:
            yield prefix
            return
        for b in bases:
            yield from _gen(prefix + b, depth - 1)

    for kmer in _gen("", k):
        vocab[kmer] = len(vocab)
    return vocab


def tokenize_kmers(sequence: str, k: int, vocab: dict) -> List[int]:
    seq = sequence.upper()
    tokens = []
    unk = vocab["<UNK>"]
    for i in range(0, len(seq) - k + 1):
        kmer = seq[i : i + k]
        tokens.append(vocab.get(kmer, unk))
    return tokens


class AttentivePooling(nn.Module):
    """Learned-weight pooling over a sequence."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class DNAEncoder(nn.Module):
    """k-mer Transformer encoder producing a single fixed-dim vector per sequence."""

    def __init__(
        self,
        k: int = 4,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        out_dim: int = 512,
        max_len: int = 4096,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.k = k
        self.vocab = kmer_vocab(k)
        self.vocab_size = len(self.vocab)
        self.max_len = max_len
        self.d_model = d_model

        self.token_embedding = nn.Embedding(
            self.vocab_size, d_model, padding_idx=self.vocab["<PAD>"]
        )
        self.position_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = AttentivePooling(d_model)
        self.proj = nn.Linear(d_model, out_dim)
        self.out_dim = out_dim

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        toks = tokenize_kmers(sequence, self.k, self.vocab)
        if len(toks) > self.max_len:
            toks = toks[: self.max_len]
        return torch.tensor(toks, dtype=torch.long)

    def collate(self, sequences: List[str]):
        tokenized = [self.encode_sequence(s) for s in sequences]
        max_l = max((t.size(0) for t in tokenized), default=1)
        max_l = max(max_l, 1)
        tokens = torch.full((len(tokenized), max_l), self.vocab["<PAD>"], dtype=torch.long)
        mask = torch.zeros((len(tokenized), max_l), dtype=torch.bool)
        for i, t in enumerate(tokenized):
            tokens[i, : t.size(0)] = t
            mask[i, : t.size(0)] = True
        return tokens, mask

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L = tokens.shape
        if L > self.max_len:
            raise ValueError(f"sequence too long: {L} > max_len {self.max_len}")
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.token_embedding(tokens) + self.position_embedding(pos_ids)
        if mask is not None:
            key_padding_mask = ~mask
        else:
            key_padding_mask = tokens.eq(self.vocab["<PAD>"])
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = self.pool(x, mask=mask)
        return self.proj(pooled)


def kmer_frequency_spectrum(sequence: str, k: int = 4) -> torch.Tensor:
    """Normalized k-mer frequency spectrum — 256-dim for k=4 (Sect. 2.9)."""
    bases = ["A", "C", "G", "T"]
    base_to_idx = {b: i for i, b in enumerate(bases)}
    n_kmers = 4 ** k
    counts = torch.zeros(n_kmers, dtype=torch.float32)
    seq = sequence.upper()
    for i in range(0, len(seq) - k + 1):
        kmer = seq[i : i + k]
        idx = 0
        valid = True
        for ch in kmer:
            if ch not in base_to_idx:
                valid = False
                break
            idx = idx * 4 + base_to_idx[ch]
        if valid:
            counts[idx] += 1.0
    total = counts.sum()
    if total > 0:
        counts = counts / total
    return counts
