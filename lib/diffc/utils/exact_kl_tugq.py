# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional
import math
import torch


def gaussian_kl_bits(q_mu: torch.Tensor, q_std: torch.Tensor, p_mu: torch.Tensor, p_std: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    q_var = torch.clamp(q_std.float(), min=1e-8).pow(2)
    p_var = torch.clamp(p_std.float(), min=1e-8).pow(2)
    kl_nat = torch.log(torch.sqrt(p_var) / torch.sqrt(q_var)) + (q_var + (q_mu.float() - p_mu.float()).pow(2)) / (2.0 * p_var) - 0.5
    kl_bits = kl_nat / math.log(2.0)
    if reduce == "sum":
        return kl_bits.sum()
    if reduce == "mean":
        return kl_bits.mean()
    return kl_bits


def kl_spatial_bits(q_mu: torch.Tensor, q_std: torch.Tensor, p_mu: torch.Tensor, p_std: torch.Tensor) -> torch.Tensor:
    kl = gaussian_kl_bits(q_mu, q_std, p_mu, p_std, reduce="none")
    while kl.dim() > 4:
        kl = kl.sum(dim=1)
    if kl.dim() == 4:
        return kl.sum(dim=1, keepdim=True)
    return kl


def masked_spatial_mean(x: torch.Tensor, masks: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    # x [B,1,H,W], masks [16,H,W], weights [B,16]
    b = x.shape[0]
    m = masks.to(device=x.device, dtype=x.dtype).view(16, -1)
    xv = x.view(b, -1)
    denom = m.sum(dim=1).clamp_min(1e-6)
    vals = torch.matmul(xv, m.t()) / denom.unsqueeze(0)
    if weights is not None:
        w = weights.to(device=x.device, dtype=x.dtype)
        return (vals * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)
    return vals.mean(dim=1)
