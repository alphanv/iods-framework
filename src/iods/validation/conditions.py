"""C1, C2, C3 condition evaluators (Sect. 2.7)."""

from __future__ import annotations

import numpy as np
import torch

from iods.prediction.losses import info_nce_estimate
from iods.validation.phylo_null import top_k_retrieval


def evaluate_c1(surplus_a_to_b, surplus_b_to_a,
                kappa_values=(1.5, 2.0, 2.5, 3.0),
                info_nce_a_b=None, info_nce_b_a=None) -> dict:
    eps = 1e-8
    if abs(surplus_a_to_b) < eps and abs(surplus_b_to_a) < eps:
        ratio = float("nan")
    elif abs(surplus_b_to_a) < eps:
        ratio = float("inf")
    else:
        ratio = surplus_a_to_b / surplus_b_to_a

    pass_kappa = {}
    for k in kappa_values:
        if np.isnan(ratio):
            pass_kappa[k] = False
        else:
            pass_kappa[k] = (1.0 / k) <= ratio <= k

    out = {
        "surplus_a_to_b": surplus_a_to_b,
        "surplus_b_to_a": surplus_b_to_a,
        "ratio": ratio,
        "pass_at_kappa": pass_kappa,
    }
    if info_nce_a_b is not None and info_nce_b_a is not None:
        out["info_nce_a_b"] = info_nce_a_b
        out["info_nce_b_a"] = info_nce_b_a
        out["info_nce_asymmetry"] = abs(info_nce_a_b - info_nce_b_a)
    return out


def evaluate_c2(iods_pred, null_pred, targets, k: int = 5, delta: float = 0.05) -> dict:
    iods_acc = top_k_retrieval(iods_pred, targets, k=k)
    null_acc = top_k_retrieval(null_pred, targets, k=k)
    surplus = iods_acc - null_acc
    return {
        "iods_topk": iods_acc, "null_topk": null_acc,
        "surplus": surplus, "delta_threshold": delta,
        "passes": surplus > delta, "k": k,
    }


def evaluate_c3(pred_with_context, pred_without_context, targets,
                k: int = 5, delta_context: float = 0.05) -> dict:
    with_acc = top_k_retrieval(pred_with_context, targets, k=k)
    without_acc = top_k_retrieval(pred_without_context, targets, k=k)
    improvement = with_acc - without_acc
    return {
        "with_context_topk": with_acc,
        "without_context_topk": without_acc,
        "context_improvement": improvement,
        "delta_threshold": delta_context,
        "passes": improvement > delta_context,
        "k": k,
    }


def evaluate_info_nce_symmetry(embeddings_a, embeddings_b, temperature: float = 0.07) -> dict:
    with torch.no_grad():
        i_ab = info_nce_estimate(embeddings_a, embeddings_b, temperature=temperature).item()
        i_ba = info_nce_estimate(embeddings_b, embeddings_a, temperature=temperature).item()
    return {
        "info_nce_a_b": i_ab, "info_nce_b_a": i_ba,
        "asymmetry": abs(i_ab - i_ba),
    }
