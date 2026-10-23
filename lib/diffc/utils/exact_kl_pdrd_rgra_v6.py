# -*- coding: utf-8 -*-
"""
exact_kl_pdrd_rgra_v6.py

Full diagonal-Gaussian KL utilities for PC-RD-RGRA-v6.
The code intentionally keeps the full KL formula even when the current DiffC
posterior uses identical q/p variances. This makes the training objective match
DiffC/RCC coding cost and remains correct if a future model predicts variance.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def gaussian_kl_bits(
    mu_q: torch.Tensor,
    std_q: torch.Tensor,
    mu_p: torch.Tensor,
    std_p: torch.Tensor,
    eps: float = 1e-12,
    reduce: str = "sum",
) -> torch.Tensor:
    """
    KL[N(mu_q, std_q^2) || N(mu_p, std_p^2)] in bits.

    Args:
        reduce: "none" returns elementwise KL, "sum" returns [B] total bits,
                "mean" returns scalar mean over all elements.
    """
    mu_q_f = mu_q.float()
    mu_p_f = mu_p.float()
    std_q_f = torch.clamp(torch.as_tensor(std_q, device=mu_q.device).float(), min=eps)
    std_p_f = torch.clamp(torch.as_tensor(std_p, device=mu_q.device).float(), min=eps)
    while std_q_f.ndim < mu_q_f.ndim:
        std_q_f = std_q_f.view(*std_q_f.shape, *([1] * (mu_q_f.ndim - std_q_f.ndim)))
    while std_p_f.ndim < mu_p_f.ndim:
        std_p_f = std_p_f.view(*std_p_f.shape, *([1] * (mu_p_f.ndim - std_p_f.ndim)))
    var_q = std_q_f.pow(2)
    var_p = std_p_f.pow(2)
    kl_nats = torch.log(std_p_f / std_q_f) + (var_q + (mu_q_f - mu_p_f).pow(2)) / (2.0 * var_p) - 0.5
    kl_bits = kl_nats / math.log(2.0)
    kl_bits = torch.nan_to_num(kl_bits, nan=0.0, posinf=1e6, neginf=0.0)
    if reduce == "none":
        return kl_bits
    if reduce == "sum":
        return kl_bits.flatten(1).sum(dim=1)
    if reduce == "mean":
        return kl_bits.mean()
    raise ValueError(f"Unsupported reduce={reduce}")


def kl_spatial_bits(
    mu_q: torch.Tensor,
    std_q: torch.Tensor,
    mu_p: torch.Tensor,
    std_p: torch.Tensor,
) -> torch.Tensor:
    """Return spatial KL map [B,H,W] in bits, summed over latent channels."""
    elem = gaussian_kl_bits(mu_q, std_q, mu_p, std_p, reduce="none")
    if elem.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] KL tensor, got shape={tuple(elem.shape)}")
    return elem.sum(dim=1)


def bpp_from_bits(bits: torch.Tensor, raw_hw: Tuple[int, int]) -> torch.Tensor:
    raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
    denom = float(max(raw_h * raw_w, 1))
    return bits.float() / denom


def masked_spatial_mean(spatial: torch.Tensor, masks: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Region-weighted mean over spatial KL/error map.

    Args:
        spatial: [B,H,W]
        masks: [16,H,W], usually soft 4x4 masks.
        weights: [B,16] or [16]
    """
    if spatial.ndim != 3:
        raise ValueError(f"spatial must be [B,H,W], got {tuple(spatial.shape)}")
    B, H, W = spatial.shape
    masks = masks.to(device=spatial.device, dtype=spatial.dtype)
    if masks.shape[-2:] != (H, W):
        import torch.nn.functional as F
        masks = F.interpolate(masks.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
    denom = torch.clamp(masks.sum(dim=(1, 2)), min=1e-6)  # [16]
    per_region = (spatial[:, None] * masks[None]).sum(dim=(2, 3)) / denom[None]
    if weights is None:
        return per_region.mean()
    weights = weights.to(device=spatial.device, dtype=spatial.dtype)
    if weights.ndim == 1:
        weights = weights.unsqueeze(0).expand(B, -1)
    return (per_region * weights).sum() / torch.clamp(weights.sum(), min=1e-6)
