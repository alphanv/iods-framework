"""C1, C2, C3 evaluation — Section 2.7, 4.6."""

import torch
import numpy as np
from typing import Dict


def evaluate_c1(surplus_ab: float, surplus_ba: float, kappa: float = 2.5) -> Dict:
    ratio = surplus_ab / max(surplus_ba, 1e-8)
    passed = (1 / kappa) <= ratio <= kappa
    return {"ratio": ratio, "kappa": kappa, "passed": passed}


def evaluate_c2(acc_iods: float, acc_phylo: float, delta: float = 0.0) -> Dict:
    surplus = acc_iods - acc_phylo
    passed = surplus > delta
    return {"surplus": surplus, "delta": delta, "passed": passed}


def evaluate_c3(acc_with_context: float, acc_without: float, threshold: float = 0.05) -> Dict:
    delta = acc_with_context - acc_without
    passed = delta > threshold
    return {"delta": delta, "threshold": threshold, "passed": passed}


def top_k_retrieval(pred_embeds, true_embeds, k: int = 5) -> float:
    """Top-k retrieval accuracy for inverse translation evaluation."""
    sims = torch.nn.functional.cosine_similarity(
        pred_embeds.unsqueeze(1), true_embeds.unsqueeze(0), dim=-1
    )
    _, topk = sims.topk(k, dim=1)
    correct = (topk == torch.arange(len(pred_embeds)).unsqueeze(1).to(topk.device)).any(dim=1)
    return correct.float().mean().item()
