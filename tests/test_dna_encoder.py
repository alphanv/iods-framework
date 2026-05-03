"""Tests for DNA encoder and k-mer utilities."""

import pytest
import torch

from iods.encoders.dna_encoder import (
    DNAEncoder, kmer_frequency_spectrum, kmer_vocab, tokenize_kmers,
)


def test_kmer_vocab_size():
    for k in (1, 2, 3, 4, 5):
        v = kmer_vocab(k)
        assert len(v) == 4 ** k + 2
        assert v["<PAD>"] == 0
        assert v["<UNK>"] == 1


def test_kmer_vocab_invalid_k():
    with pytest.raises(ValueError):
        kmer_vocab(0)
    with pytest.raises(ValueError):
        kmer_vocab(9)


def test_tokenize_basic():
    v = kmer_vocab(3)
    seq = "ACGTACG"
    tokens = tokenize_kmers(seq, 3, v)
    assert len(tokens) == len(seq) - 3 + 1
    assert all(t != v["<UNK>"] for t in tokens)


def test_tokenize_with_n():
    v = kmer_vocab(3)
    seq = "ACGNTAC"
    tokens = tokenize_kmers(seq, 3, v)
    # 3-mers: ACG, CGN, GNT, NTA, TAC. Three contain N -> UNK.
    assert tokens.count(v["<UNK>"]) == 3
    assert tokens.count(v["<UNK>"]) < len(tokens)


def test_tokenize_lowercase():
    v = kmer_vocab(3)
    upper = tokenize_kmers("ACGTACG", 3, v)
    lower = tokenize_kmers("acgtacg", 3, v)
    assert upper == lower


def test_kmer_spectrum_dimension():
    spec = kmer_frequency_spectrum("ACGTACGTACGT", k=4)
    assert spec.shape == (256,)
    assert torch.isclose(spec.sum(), torch.tensor(1.0), atol=1e-6)


def test_kmer_spectrum_known_value():
    spec = kmer_frequency_spectrum("AAAAAAAA", k=4)
    assert spec[0].item() == pytest.approx(1.0)
    assert spec[1:].sum().item() == pytest.approx(0.0)


def test_kmer_spectrum_handles_empty():
    spec = kmer_frequency_spectrum("AC", k=4)
    assert spec.shape == (256,)
    assert spec.sum().item() == 0.0


def test_dna_encoder_runs():
    enc = DNAEncoder(k=3, d_model=32, n_heads=2, n_layers=1, out_dim=64, max_len=128)
    sequences = ["ACGTACGTACGT", "GGCCAATT", "TATATATA"]
    tokens, mask = enc.collate(sequences)
    assert tokens.shape[0] == 3
    assert mask.shape == tokens.shape
    out = enc(tokens, mask=mask)
    assert out.shape == (3, 64)
    assert torch.all(torch.isfinite(out))


def test_dna_encoder_gradient_flows():
    enc = DNAEncoder(k=3, d_model=32, n_heads=2, n_layers=1, out_dim=64, max_len=128)
    tokens, mask = enc.collate(["ACGTACGTACGT", "GGCCAATT"])
    out = enc(tokens, mask=mask)
    loss = out.sum()
    loss.backward()
    assert enc.token_embedding.weight.grad is not None
    assert torch.any(enc.token_embedding.weight.grad != 0)


def test_dna_encoder_handles_padding():
    enc = DNAEncoder(k=3, d_model=32, n_heads=2, n_layers=1, out_dim=64, max_len=128)
    tokens, mask = enc.collate(["ACGTACGT", "AAAA"])
    enc.eval()
    with torch.no_grad():
        out = enc(tokens, mask=mask)
    assert out.shape == (2, 64)
    assert torch.all(torch.isfinite(out))
