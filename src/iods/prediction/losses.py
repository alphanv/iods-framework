"""Loss functions per [F7] and Sect. 2.10."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def heteroskedastic_nll(mu, log_sigma2, target):
    """L_NLL = sum_d [log sigma^2_d + (y_d - mu_d)^2 / sigma^2_d]"""
    sigma2 = torch.exp(log_sigma2)
    return (log_sigma2 + (target - mu) ** 2 / sigma2).mean()


def cosine_distance(a, b):
    """d_DNA = 1 - cos(a, b)."""
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()


def kmer_kl_divergence(pred_logits, target_freq, eps: float = 1e-8):
    pred_log_prob = F.log_softmax(pred_logits, dim=-1)
    target = target_freq.clamp(min=eps)
    return F.kl_div(pred_log_prob, target, reduction="batchmean")


def pfam_bce(pred_logits, target):
    return F.binary_cross_entropy_with_logits(pred_logits, target.float())


def info_nce_loss(a, b, temperature: float = 0.07):
    """Symmetric InfoNCE."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = (a @ b.t()) / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )


def info_nce_estimate(a, b, temperature: float = 0.07):
    """MI lower bound: I(a; b) >= log(N) - L_InfoNCE."""
    loss = info_nce_loss(a, b, temperature=temperature)
    log_N = torch.log(torch.tensor(float(a.size(0)), device=a.device))
    return log_N - loss


def total_loss(forward_loss, inverse_loss, mi_loss, reg_loss,
               lambda_inv: float = 1.0, lambda_mi: float = 0.05,
               lambda_reg: float = 0.001):
    return (forward_loss
            + lambda_inv * inverse_loss
            + lambda_mi * mi_loss
            + lambda_reg * reg_loss)
