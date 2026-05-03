"""Tests for the magnitude function (formula F1)."""

import pytest
import torch

from iods.magnitude import (
    ContextEmbedding, MagnitudeFunction, MagnitudeFunctionSoftmax, apply_magnitude,
)


@pytest.fixture
def context_embedding():
    return ContextEmbedding(
        n_env_categories=4, n_phase_categories=4,
        d_time=32, d_env=16, d_phase=16,
    )


@pytest.fixture
def magnitude(context_embedding):
    return MagnitudeFunction(n_modalities=7, context_embedding=context_embedding)


def _make_context(B=8):
    t = torch.linspace(0.0, 10.0, B)
    e = torch.randint(0, 4, (B,))
    s = torch.randint(0, 4, (B,))
    return t, e, s


def test_magnitude_in_unit_interval(magnitude):
    t, e, s = _make_context(B=16)
    m = magnitude(t, e, s)
    assert m.shape == (16, 7)
    assert torch.all(m > 0)
    assert torch.all(m < 1)


def test_magnitude_with_zero_context(context_embedding):
    mag = MagnitudeFunction(n_modalities=4, context_embedding=context_embedding)
    with torch.no_grad():
        mag.w.zero_()
        mag.b.zero_()
    t, e, s = _make_context(B=4)
    m = mag(t, e, s)
    assert torch.allclose(m, torch.full_like(m, 0.5), atol=1e-6)


def test_sigmoid_gradient_form(context_embedding):
    mag = MagnitudeFunction(n_modalities=3, context_embedding=context_embedding)
    t = torch.tensor([1.0, 2.0, 3.0])
    e = torch.tensor([0, 1, 2])
    s = torch.tensor([2, 1, 0])
    m = mag(t, e, s)
    c = mag.context_embedding(t, e, s)
    logits = c @ mag.w.t() + mag.b
    expected = torch.sigmoid(logits)
    assert torch.allclose(m, expected, atol=1e-6)
    d = expected * (1 - expected)
    assert torch.all(d > 0)
    assert torch.all(d <= 0.25 + 1e-6)


def test_renormalize_for_presence_zeros_absent(magnitude):
    t, e, s = _make_context(B=4)
    m = magnitude(t, e, s)
    presence = torch.tensor(
        [[1, 1, 0, 1, 1, 1, 0],
         [1, 0, 0, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 0, 1],
         [0, 1, 1, 0, 1, 1, 1]],
        dtype=torch.float32,
    )
    m_renorm = magnitude.renormalize_for_presence(m, presence)
    assert torch.all((m_renorm * (1 - presence)) == 0)
    assert torch.all((m_renorm * presence)[presence == 1] > 0)


def test_renormalize_preserves_sum(magnitude):
    t, e, s = _make_context(B=4)
    m = magnitude(t, e, s)
    presence = torch.tensor(
        [[1, 1, 0, 1, 1, 1, 0],
         [1, 0, 0, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 0, 1],
         [0, 1, 1, 0, 1, 1, 1]],
        dtype=torch.float32,
    )
    sum_before = m.sum(dim=-1)
    m_renorm = magnitude.renormalize_for_presence(m, presence)
    sum_after = m_renorm.sum(dim=-1)
    assert torch.allclose(sum_before, sum_after, atol=1e-5)


def test_renormalize_all_present_is_identity(magnitude):
    t, e, s = _make_context(B=4)
    m = magnitude(t, e, s)
    presence = torch.ones_like(m)
    m_renorm = magnitude.renormalize_for_presence(m, presence)
    assert torch.allclose(m_renorm, m, atol=1e-6)


def test_apply_magnitude_shapes():
    B, M, D = 4, 7, 16
    emb = torch.randn(B, M, D)
    mag = torch.rand(B, M)
    out = apply_magnitude(emb, mag)
    assert out.shape == (B, M, D)


def test_apply_magnitude_scaling():
    B, M, D = 2, 3, 4
    emb = torch.ones(B, M, D)
    mag = torch.tensor([[0.5, 1.0, 0.0], [1.0, 0.0, 0.5]])
    out = apply_magnitude(emb, mag)
    expected = torch.zeros_like(emb)
    expected[0, 0, :] = 0.5; expected[0, 1, :] = 1.0; expected[0, 2, :] = 0.0
    expected[1, 0, :] = 1.0; expected[1, 1, :] = 0.0; expected[1, 2, :] = 0.5
    assert torch.allclose(out, expected)


def test_apply_magnitude_shape_validation():
    with pytest.raises(ValueError):
        apply_magnitude(torch.randn(4, 7), torch.randn(4, 7))
    with pytest.raises(ValueError):
        apply_magnitude(torch.randn(4, 7, 16), torch.randn(4, 8))


def test_softmax_variant_sums_to_one(context_embedding):
    mag = MagnitudeFunctionSoftmax(n_modalities=5, context_embedding=context_embedding)
    t, e, s = _make_context(B=4)
    m = mag(t, e, s)
    sums = m.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_magnitude_deterministic(magnitude):
    t, e, s = _make_context(B=4)
    magnitude.eval()
    m1 = magnitude(t, e, s)
    m2 = magnitude(t, e, s)
    assert torch.allclose(m1, m2)
