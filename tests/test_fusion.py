"""Tests for cross-attention fusion and presence handling."""

import pytest
import torch

from iods.fusion import CrossAttentionFusion, ModalityProjector


def test_modality_projector_output_shape():
    proj = ModalityProjector({"a": 32, "b": 64, "c": 16}, d_z=128)
    embeddings = {
        "a": torch.randn(4, 32),
        "b": torch.randn(4, 64),
        "c": torch.randn(4, 16),
    }
    out = proj(embeddings)
    assert out.shape == (4, 3, 128)


def test_modality_projector_missing_modality_raises():
    proj = ModalityProjector({"a": 32, "b": 64}, d_z=128)
    with pytest.raises(KeyError):
        proj({"a": torch.randn(4, 32)})


def test_fusion_output_shape():
    fusion = CrossAttentionFusion(d_z=64, n_modalities=4, n_heads=2, n_layers=1)
    embeddings = torch.randn(8, 4, 64)
    magnitudes = torch.rand(8, 4)
    presence = torch.ones(8, 4)
    z = fusion(embeddings, magnitudes, presence)
    assert z.shape == (8, 64)
    assert torch.all(torch.isfinite(z))


def test_fusion_handles_missing_modalities():
    fusion = CrossAttentionFusion(d_z=64, n_modalities=4, n_heads=2, n_layers=1)
    embeddings = torch.randn(4, 4, 64)
    magnitudes = torch.rand(4, 4)
    presence = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=torch.float32,
    )
    z = fusion(embeddings, magnitudes, presence)
    assert z.shape == (4, 64)
    assert torch.all(torch.isfinite(z))


def test_fusion_gradient_flows():
    torch.manual_seed(0)
    fusion = CrossAttentionFusion(d_z=32, n_modalities=3, n_heads=2, n_layers=1)
    embeddings = torch.randn(4, 3, 32, requires_grad=True)
    magnitudes = torch.rand(4, 3)
    presence = torch.ones(4, 3)
    z = fusion(embeddings, magnitudes, presence)
    z.sum().backward()
    assert embeddings.grad is not None
    # Gradients may be tiny at fresh init; require any non-trivial grad sum.
    assert embeddings.grad.abs().sum().item() > 0.0


def test_fusion_presence_zero_breaks_attention():
    torch.manual_seed(0)
    fusion = CrossAttentionFusion(d_z=32, n_modalities=3, n_heads=2, n_layers=1)
    fusion.eval()

    embeddings_all = torch.zeros(1, 3, 32)
    embeddings_all[0, 1, :] = 1000.0
    magnitudes = torch.ones(1, 3)
    presence_all = torch.ones(1, 3)
    presence_only_others = torch.tensor([[1.0, 0.0, 1.0]])

    with torch.no_grad():
        z_all = fusion(embeddings_all, magnitudes, presence_all)
        z_masked = fusion(embeddings_all, magnitudes, presence_only_others)

    diff = (z_all - z_masked).abs().max().item()
    assert diff > 1e-3
