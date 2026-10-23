# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from lib.diffc.utils.alpha_beta import get_alpha_prod_and_beta_prod
from lib.diffc.utils.p import P
from lib.diffc.utils.q import Q

LOG2_DUP_CMDC = 0.6931471805599453


def _crop_spec(latent: torch.Tensor, crop_size: int, seed: int):
    h, w = int(latent.shape[-2]), int(latent.shape[-1])
    ch = min(max(1, int(crop_size)), h)
    cw = min(max(1, int(crop_size)), w)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    top = 0 if ch == h else int(torch.randint(0, h - ch + 1, (1,), generator=gen).item())
    left = 0 if cw == w else int(torch.randint(0, w - cw + 1, (1,), generator=gen).item())
    return top, left, ch, cw


def _crop_latent(latent: torch.Tensor, spec):
    top, left, h, w = spec
    return latent[..., top:top + h, left:left + w]


def _charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred.float() - target.float()).pow(2) + float(eps) ** 2).mean()


def _gradient_charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    px = pred[..., :, 1:] - pred[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    py = pred[..., 1:, :] - pred[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * (_charbonnier(px, tx, eps) + _charbonnier(py, ty, eps))


def _gray3(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    g = x.mean(dim=1, keepdim=True)
    return g.repeat(1, 3, 1, 1)


def decoded_patch_quality_loss(
    *, model, z0: torch.Tensor, x0_base: torch.Tensor, x0_correct: torch.Tensor,
    modality: str, crop_size: int, seed: int, edge_weight: float,
    lpips_model=None, lpips_weight: float = 0.0,
):

    spec = _crop_spec(z0, crop_size, seed)
    z0_crop = _crop_latent(z0.detach(), spec)
    base_crop = _crop_latent(x0_base.detach(), spec)
    corr_crop = _crop_latent(x0_correct, spec)
    with torch.no_grad():
        target_img = model.decode_latent_for_loss(z0_crop)
        base_img = model.decode_latent_for_loss(base_crop)
    corr_img = model.decode_latent_for_loss(corr_crop)
    is_ir = str(modality).lower() in {"thermal", "ir", "t"}
    if is_ir:
        target_img = _gray3(target_img)
        base_img = _gray3(base_img)
        corr_img = _gray3(corr_img)
    base_char = _charbonnier(base_img, target_img).detach()
    corr_char = _charbonnier(corr_img, target_img)
    char_ratio = corr_char / torch.clamp(base_char, min=1e-7)
    base_edge = _gradient_charbonnier(base_img, target_img).detach()
    corr_edge = _gradient_charbonnier(corr_img, target_img)
    edge_ratio = corr_edge / torch.clamp(base_edge, min=1e-7)
    lpips_ratio = torch.ones((), device=z0.device)
    total = char_ratio + float(edge_weight) * edge_ratio
    if lpips_model is not None and (not is_ir) and float(lpips_weight) > 0.0:
        with torch.no_grad():
            base_lp = lpips_model(base_img, target_img).detach()
        corr_lp = lpips_model(corr_img, target_img)
        lpips_ratio = corr_lp / torch.clamp(base_lp, min=1e-7)
        total = total + float(lpips_weight) * lpips_ratio
    return total, {
        "patch_quality": total.detach(),
        "patch_char_ratio": char_ratio.detach(),
        "patch_edge_ratio": edge_ratio.detach(),
        "patch_lpips_ratio": lpips_ratio.detach(),
    }


def gaussian_kl_bits(mean_q: torch.Tensor, std_q: torch.Tensor, mean_p: torch.Tensor, std_p: torch.Tensor) -> torch.Tensor:
    var_q = torch.clamp(std_q.float().pow(2), min=1e-12)
    var_p = torch.clamp(std_p.float().pow(2), min=1e-12)
    kl_nats = torch.log(torch.sqrt(var_p) / torch.sqrt(var_q)) + (var_q + (mean_q.float() - mean_p.float()).pow(2)) / (2.0 * var_p) - 0.5
    return kl_nats.sum() / LOG2_DUP_CMDC


def make_noisy_latent(z0: torch.Tensor, current_snr: torch.Tensor, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=z0.device)
    gen.manual_seed(int(seed))
    eps = torch.randn(z0.shape, generator=gen, device=z0.device, dtype=torch.float32)
    alpha, beta = get_alpha_prod_and_beta_prod(current_snr.float())
    noisy = torch.sqrt(alpha).to(z0.device) * z0.float() + torch.sqrt(beta).to(z0.device) * eps
    return noisy.to(dtype=z0.dtype), eps


def exact_edge_forward(
    model,
    noisy_latent: torch.Tensor,
    z0: torch.Tensor,
    current_t: int,
    next_t: int,
    codes: Optional[Mapping[str, str]],
    modality: str,
    trainable: bool,
    condition_mode: str = "full",
):


    
    
    if codes is None:
        model.set_condition(None, modality, enabled=False, condition_mode="full")
    else:
        model.set_condition(codes, modality, enabled=True, condition_mode=condition_mode)
    current_snr = model.get_timestep_snr(int(current_t)).to(model.device)
    next_snr = model.get_timestep_snr(int(next_t)).to(model.device)
    if trainable:
        noise_pred = model.predict_noise(noisy_latent.to(dtype=model.dtype), int(current_t)).float()
    else:
        with torch.no_grad():
            noise_pred = model.predict_noise(noisy_latent.to(dtype=model.dtype), int(current_t)).float()
    p_mu, std = P(noisy_latent.float(), noise_pred.float(), current_snr, next_snr)
    q_mu = Q(noisy_latent.float(), z0.float(), current_snr, next_snr)
    kl_bits = gaussian_kl_bits(q_mu, std, p_mu, std)
    alpha, beta = get_alpha_prod_and_beta_prod(current_snr.float())
    x0_hat = (noisy_latent.float() - torch.sqrt(beta).to(model.device) * noise_pred) / torch.sqrt(torch.clamp(alpha, min=1e-8)).to(model.device)
    return kl_bits, x0_hat, noise_pred


def compute_losses(
    *,
    model,
    z0: torch.Tensor,
    noisy: torch.Tensor,
    current_t: int,
    next_t: int,
    correct_codes: Mapping[str, str],
    negative_codes: Optional[Mapping[str, str]],
    modality: str,
    num_pixels: int,
    condition_mode: str = "full",
    bridge_gain_reference: float = 0.02,
    bridge_dist_margin_rel: float = 0.002,
    specificity_margin_rel: float = 0.005,
    leak_allow_rel: float = 0.005,
) -> Dict[str, torch.Tensor]:
    eps = 1e-9
    kl_base_bits, x0_base, _ = exact_edge_forward(
        model, noisy, z0, current_t, next_t, None, modality, trainable=False
    )
    kl_corr_bits, x0_corr, _ = exact_edge_forward(
        model, noisy, z0, current_t, next_t, correct_codes, modality, trainable=True,
        condition_mode=condition_mode
    )
    denom = float(max(1, int(num_pixels)))
    base_bpp = kl_base_bits.detach() / denom
    corr_bpp = kl_corr_bits / denom
    rho = corr_bpp / torch.clamp(base_bpp, min=eps)
    path_loss = rho

    base_dist = F.smooth_l1_loss(x0_base.detach().float(), z0.detach().float(), beta=0.05)
    corr_dist = F.smooth_l1_loss(x0_corr.float(), z0.float(), beta=0.05)
    dist_ratio = corr_dist / torch.clamp(base_dist.detach(), min=eps)
    dist_loss = dist_ratio

    kl_gain_rel = 1.0 - rho
    bridge_weight = torch.clamp(kl_gain_rel.detach() / float(bridge_gain_reference), min=0.0, max=1.0)
    bridge_loss = bridge_weight * torch.relu(dist_ratio - (1.0 - float(bridge_dist_margin_rel)))

    specificity_loss = torch.zeros((), device=z0.device)
    contrast_loss = torch.zeros((), device=z0.device)
    leak_loss = torch.zeros((), device=z0.device)
    neg_bpp = base_bpp.detach()
    neg_over_base = torch.ones((), device=z0.device)
    correct_over_neg = torch.ones((), device=z0.device)
    if negative_codes is not None:
        kl_neg_bits, _, _ = exact_edge_forward(
            model, noisy.detach(), z0.detach(), current_t, next_t, negative_codes, modality, trainable=True,
            condition_mode=condition_mode
        )
        neg_bpp = kl_neg_bits / denom
        correct_over_neg = corr_bpp / torch.clamp(neg_bpp, min=eps)
        contrast_loss = torch.relu(correct_over_neg - (1.0 - float(specificity_margin_rel)))
        neg_over_base = neg_bpp / torch.clamp(base_bpp, min=eps)
        negative_gain_rel = 1.0 - neg_over_base
        leak_loss = torch.relu(negative_gain_rel - float(leak_allow_rel))
        specificity_loss = contrast_loss + 0.5 * leak_loss

    bridge_success = ((rho.detach() < 1.0) & (dist_ratio.detach() < 1.0)).float()
    aux = model.get_last_aux()
    return {
        "path": path_loss,
        "dist": dist_loss,
        "bridge": bridge_loss,
        "spec": specificity_loss,
        "contrast": contrast_loss.detach(),
        "leak": leak_loss.detach(),
        "base_kl_bpp": base_bpp.detach(),
        "correct_kl_bpp": corr_bpp.detach(),
        "negative_kl_bpp": neg_bpp.detach(),
        "rho": rho.detach(),
        "kl_gain_rel": kl_gain_rel.detach(),
        "base_dist": base_dist.detach(),
        "correct_dist": corr_dist.detach(),
        "dist_ratio": dist_ratio.detach(),
        "bridge_weight": bridge_weight.detach(),
        "bridge_success": bridge_success.detach(),
        "correct_over_negative": correct_over_neg.detach(),
        "negative_over_base": neg_over_base.detach(),
        "gate_attn": torch.tensor(float(aux.get("gate_attn", 0.0)), device=z0.device),
        "gate_film": torch.tensor(float(aux.get("gate_film", 0.0)), device=z0.device),
        "gate_tsp": torch.tensor(float(aux.get("gate_tsp", 0.0)), device=z0.device),
        "gate_spp": torch.tensor(float(aux.get("gate_spp", 0.0)), device=z0.device),
        "gate_delta": torch.tensor(float(aux.get("gate_delta", 0.0)), device=z0.device),
    }


def total_loss(parts: Dict[str, torch.Tensor], model, lambda_path: float, lambda_dist: float, lambda_bridge: float, lambda_spec: float, lambda_reg: float, condition_mode: str = "full"):
    reg = model.gate_regularization(condition_mode) + 0.5 * model.delta_regularization(condition_mode)
    total = (
        float(lambda_path) * parts["path"]
        + float(lambda_dist) * parts["dist"]
        + float(lambda_bridge) * parts["bridge"]
        + float(lambda_spec) * parts["spec"]
        + float(lambda_reg) * reg
    )
    return total, reg


def backward_losses_memory_efficient(
    *,
    model,
    z0: torch.Tensor,
    noisy: torch.Tensor,
    current_t: int,
    next_t: int,
    correct_codes: Mapping[str, str],
    negative_codes: Optional[Mapping[str, str]],
    modality: str,
    num_pixels: int,
    condition_mode: str,
    loss_scale: float,
    lambda_path: float,
    lambda_dist: float,
    lambda_bridge: float,
    lambda_spec: float,
    lambda_reg: float,
    bridge_gain_reference: float = 0.02,
    bridge_dist_margin_rel: float = 0.002,
    specificity_margin_rel: float = 0.005,
    leak_allow_rel: float = 0.005,
    lambda_patch: float = 0.0,
    patch_crop_latent_size: int = 32,
    patch_seed: int = 0,
    patch_edge_weight: float = 0.20,
    patch_lpips_model=None,
    patch_lpips_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:

    eps = 1e-9
    denom = float(max(1, int(num_pixels)))

    
    kl_base_bits, x0_base, _ = exact_edge_forward(
        model, noisy, z0, current_t, next_t, None, modality, trainable=False
    )
    base_bpp = (kl_base_bits / denom).detach()
    x0_base_detached_for_patch = x0_base.detach()
    base_dist = F.smooth_l1_loss(
        x0_base_detached_for_patch.float(), z0.detach().float(), beta=0.05
    ).detach()
    del x0_base, kl_base_bits

    
    kl_corr_bits, x0_corr, _ = exact_edge_forward(
        model, noisy, z0, current_t, next_t, correct_codes, modality, trainable=True,
        condition_mode=condition_mode
    )
    corr_bpp = kl_corr_bits / denom
    rho = corr_bpp / torch.clamp(base_bpp, min=eps)
    corr_dist = F.smooth_l1_loss(x0_corr.float(), z0.float(), beta=0.05)
    dist_ratio = corr_dist / torch.clamp(base_dist, min=eps)
    kl_gain_rel = 1.0 - rho
    bridge_weight = torch.clamp(
        kl_gain_rel.detach() / float(bridge_gain_reference), min=0.0, max=1.0
    )
    bridge_loss = bridge_weight * torch.relu(
        dist_ratio - (1.0 - float(bridge_dist_margin_rel))
    )
    patch_loss = torch.zeros((), device=z0.device)
    patch_diag = {}
    if float(lambda_patch) > 0.0:
        patch_loss, patch_diag = decoded_patch_quality_loss(
            model=model, z0=z0, x0_base=x0_base_detached_for_patch, x0_correct=x0_corr,
            modality=modality, crop_size=int(patch_crop_latent_size), seed=int(patch_seed),
            edge_weight=float(patch_edge_weight), lpips_model=patch_lpips_model,
            lpips_weight=float(patch_lpips_weight),
        )
    reg = model.gate_regularization(condition_mode) + 0.5 * model.delta_regularization(condition_mode)
    positive_total = (
        float(lambda_path) * rho
        + float(lambda_dist) * dist_ratio
        + float(lambda_bridge) * bridge_loss
        + float(lambda_patch) * patch_loss
        + float(lambda_reg) * reg
    )
    (float(loss_scale) * positive_total).backward()

    corr_bpp_detached = corr_bpp.detach()
    rho_detached = rho.detach()
    corr_dist_detached = corr_dist.detach()
    dist_ratio_detached = dist_ratio.detach()
    kl_gain_detached = kl_gain_rel.detach()
    bridge_weight_detached = bridge_weight.detach()
    bridge_loss_detached = bridge_loss.detach()
    reg_detached = reg.detach()
    patch_loss_detached = patch_loss.detach()
    patch_diag_detached = {k: v.detach() for k, v in patch_diag.items()}
    positive_total_detached = positive_total.detach()
    del kl_corr_bits, x0_corr, corr_bpp, rho, corr_dist, dist_ratio
    del kl_gain_rel, bridge_weight, bridge_loss, patch_loss, reg, positive_total

    specificity_loss = torch.zeros((), device=z0.device)
    contrast_loss = torch.zeros((), device=z0.device)
    leak_loss = torch.zeros((), device=z0.device)
    neg_bpp_detached = base_bpp
    correct_over_neg_detached = torch.ones((), device=z0.device)
    neg_over_base_detached = torch.ones((), device=z0.device)

    if negative_codes is not None and float(lambda_spec) > 0.0:
        kl_neg_bits, _x0_neg, _ = exact_edge_forward(
            model,
            noisy.detach(),
            z0.detach(),
            current_t,
            next_t,
            negative_codes,
            modality,
            trainable=True,
            condition_mode=condition_mode,
        )
        neg_bpp = kl_neg_bits / denom
        correct_over_neg = corr_bpp_detached / torch.clamp(neg_bpp, min=eps)
        contrast_loss = torch.relu(
            correct_over_neg - (1.0 - float(specificity_margin_rel))
        )
        neg_over_base = neg_bpp / torch.clamp(base_bpp, min=eps)
        negative_gain_rel = 1.0 - neg_over_base
        leak_loss = torch.relu(negative_gain_rel - float(leak_allow_rel))
        specificity_loss = contrast_loss + 0.5 * leak_loss
        (float(loss_scale) * float(lambda_spec) * specificity_loss).backward()

        neg_bpp_detached = neg_bpp.detach()
        correct_over_neg_detached = correct_over_neg.detach()
        neg_over_base_detached = neg_over_base.detach()
        specificity_loss_detached = specificity_loss.detach()
        contrast_loss_detached = contrast_loss.detach()
        leak_loss_detached = leak_loss.detach()
        del kl_neg_bits, _x0_neg, neg_bpp, correct_over_neg, neg_over_base
        del negative_gain_rel, specificity_loss, contrast_loss, leak_loss
    else:
        specificity_loss_detached = specificity_loss.detach()
        contrast_loss_detached = contrast_loss.detach()
        leak_loss_detached = leak_loss.detach()

    total_detached = positive_total_detached + float(lambda_spec) * specificity_loss_detached
    bridge_success = ((rho_detached < 1.0) & (dist_ratio_detached < 1.0)).float()
    aux = model.get_last_aux()
    return {
        "total": total_detached,
        "path": rho_detached,
        "dist": dist_ratio_detached,
        "bridge": bridge_loss_detached,
        "patch": patch_loss_detached,
        "spec": specificity_loss_detached,
        "contrast": contrast_loss_detached,
        "leak": leak_loss_detached,
        "reg": reg_detached,
        "base_kl_bpp": base_bpp,
        "correct_kl_bpp": corr_bpp_detached,
        "negative_kl_bpp": neg_bpp_detached,
        "rho": rho_detached,
        "kl_gain_rel": kl_gain_detached,
        "base_dist": base_dist,
        "correct_dist": corr_dist_detached,
        "dist_ratio": dist_ratio_detached,
        "bridge_weight": bridge_weight_detached,
        "bridge_success": bridge_success,
        "patch_quality": patch_diag_detached.get("patch_quality", torch.ones((), device=z0.device)),
        "patch_char_ratio": patch_diag_detached.get("patch_char_ratio", torch.ones((), device=z0.device)),
        "patch_edge_ratio": patch_diag_detached.get("patch_edge_ratio", torch.ones((), device=z0.device)),
        "patch_lpips_ratio": patch_diag_detached.get("patch_lpips_ratio", torch.ones((), device=z0.device)),
        "correct_over_negative": correct_over_neg_detached,
        "negative_over_base": neg_over_base_detached,
        "gate_attn": torch.tensor(float(aux.get("gate_attn", 0.0)), device=z0.device),
        "gate_film": torch.tensor(float(aux.get("gate_film", 0.0)), device=z0.device),
        "gate_tsp": torch.tensor(float(aux.get("gate_tsp", 0.0)), device=z0.device),
        "gate_spp": torch.tensor(float(aux.get("gate_spp", 0.0)), device=z0.device),
        "gate_delta": torch.tensor(float(aux.get("gate_delta", 0.0)), device=z0.device),
    }
