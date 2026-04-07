"""Distance functions by modality — Section 2.10."""

import torch
import torch.nn.functional as F


def d_dna(pred_embed, true_embed):
    return 1 - F.cosine_similarity(pred_embed, true_embed, dim=-1).mean()

def d_image(pred, true, ssim_weight=0.5):
    mse = F.mse_loss(pred, true)
    return mse  # Full SSIM requires torchmetrics

def d_sound(pred_spec, true_spec):
    pred_flat = pred_spec.flatten(1)
    true_flat = true_spec.flatten(1)
    cos = F.cosine_similarity(pred_flat, true_flat, dim=-1)
    return (1 - cos).mean()

def total_loss(l_forward, l_inverse, l_mi=0, lambda_inv=1.0, lambda_mi=0.01, lambda_reg=1e-4, reg=0):
    return l_forward + lambda_inv * l_inverse + lambda_mi * l_mi + lambda_reg * reg
