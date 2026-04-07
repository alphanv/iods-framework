"""C1-C3 condition testing — Section 4.6."""
from src.utils.metrics import evaluate_c1, evaluate_c2, evaluate_c3, top_k_retrieval
import torch

def test_c1_symmetric():
    r = evaluate_c1(surplus_ab=0.3, surplus_ba=0.25, kappa=2.5)
    assert r["passed"], f"C1 should pass for near-symmetric surplus, got ratio {r['ratio']}"

def test_c1_asymmetric():
    r = evaluate_c1(surplus_ab=0.5, surplus_ba=0.1, kappa=2.5)
    assert not r["passed"], "C1 should fail for asymmetric surplus"

def test_c2_surplus():
    r = evaluate_c2(acc_iods=0.75, acc_phylo=0.60, delta=0.0)
    assert r["passed"] and abs(r["surplus"] - 0.15) < 1e-9

def test_c3_context():
    r = evaluate_c3(acc_with_context=0.80, acc_without=0.70, threshold=0.05)
    assert r["passed"]

def test_top_k():
    pred = torch.randn(10, 64)
    true = pred + torch.randn(10, 64) * 0.1  # near-perfect
    acc = top_k_retrieval(pred, true, k=5)
    assert acc > 0.5, f"Top-5 retrieval should be high for near-matches, got {acc}"

if __name__ == "__main__":
    test_c1_symmetric()
    test_c1_asymmetric()
    test_c2_surplus()
    test_c3_context()
    test_top_k()
    print("All condition tests passed.")
