"""Magnitude function identifiability diagnostics — Section 2.6."""
import torch
from src.magnitude.magnitude import MagnitudeFunction

def test_sigmoid_bounds():
    mag = MagnitudeFunction(n_modalities=5, context_dim=96)
    ctx = torch.randn(16, 96)
    m = mag(ctx)
    assert (m > 0).all() and (m < 1).all(), "Magnitudes must be in (0,1)"

def test_renormalization():
    mag = MagnitudeFunction(n_modalities=3, context_dim=96)
    ctx = torch.randn(4, 96)
    m = mag(ctx)
    presence = torch.tensor([[1,1,0],[1,0,1],[0,1,1],[1,1,1]]).float()
    normed = mag.renormalize(m, presence)
    sums = normed.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_ablation_uniform():
    """Diagnostic 1: compare learned vs uniform magnitudes."""
    mag = MagnitudeFunction(n_modalities=5, context_dim=96)
    ctx = torch.randn(32, 96)
    learned = mag(ctx)
    uniform = torch.ones_like(learned) * 0.5
    assert not torch.allclose(learned, uniform, atol=0.01), "Untrained magnitudes should vary"

if __name__ == "__main__":
    test_sigmoid_bounds()
    test_renormalization()
    test_ablation_uniform()
    print("All magnitude tests passed.")
