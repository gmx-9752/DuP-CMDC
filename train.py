# -*- coding: utf-8 -*-
"""Train TriTSP-Bridge on paired RGB/raw-thermal data.

One GPU:
  CUDA_VISIBLE_DEVICES=0 python train_sana_tritsp_bridge.py ...

Two GPUs (recommended for 24GB cards):
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      train_sana_tritsp_bridge.py ...

With two processes, rank 0 trains RGB and rank 1 trains thermal on the same
pair, fixed-path edge, stage, and TSP code.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

from lib.models.SanaDiT_dup_cmdc import SanaDiTModel_DuPCmdc
from lib.dup_cmdc.distributed import (
    average_gradients,
    barrier,
    broadcast_parameters,
    cleanup_distributed,
    init_distributed,
)
from lib.dup_cmdc.image_io import load_rgb_tensor, load_thermal_tensor
from lib.dup_cmdc.losses import (
    backward_losses_memory_efficient,
    compute_losses,
    exact_edge_forward,
    make_noisy_latent,
)
from lib.dup_cmdc.negative_scns import (
    generate_counterfactual,
    negative_mode_for_epoch,
    semantic_distance,
)
from lib.dup_cmdc.stage_prompt import (
    build_path_edges_to_stop,
    representative_edges,
    split_edges_by_stage,
)
from lib.dup_cmdc.text_cache import TriTSPTextCache
from lib.dup_cmdc.pairing import build_paired_items, write_pair_manifest


DIAG_KEYS_DUP_CMDC = (
    "total", "path", "dist", "bridge", "spec", "reg", "rho", "kl_gain_rel",
    "dist_ratio", "bridge_weight", "bridge_success", "correct_over_negative",
    "negative_over_base", "base_kl_bpp", "correct_kl_bpp", "negative_kl_bpp",
    "base_dist", "correct_dist", "patch", "patch_quality", "patch_char_ratio",
    "patch_edge_ratio", "patch_lpips_ratio", "gate_attn", "gate_film",
    "gate_tsp", "gate_spp", "gate_delta",
)


def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return edict(yaml.safe_load(f))


def append_jsonl(path: Path, row: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def append_csv(path: Path, row: Mapping) -> None:
    """Append a row while preserving a stable union of CSV columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row_dict = dict(row)
    if not path.exists():
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
            writer.writeheader()
            writer.writerow(row_dict)
        return

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        old_rows = list(reader)
    new_fields = old_fields + [k for k in row_dict.keys() if k not in old_fields]
    if new_fields != old_fields:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_fields)
            writer.writeheader()
            writer.writerows(old_rows)
            writer.writerow(row_dict)
    else:
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=old_fields, extrasaction="ignore")
            writer.writerow(row_dict)


def log_event(path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{stamp}] {message}"
    print(text)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def make_model(config, args, device: str):
    return SanaDiTModel_DuPCmdc(
        model_id=config.get("model_path", config.get("model_id")),
        device=device,
        transformer_dtype=config.get("transformer_dtype", "float16"),
        text_vae_dtype=config.get("text_vae_dtype", "bfloat16"),
        local_files_only=config.get("local_files_only", True),
        max_sequence_length=config.get("max_sequence_length", 300),
        tritsp_text_cache_path=args.text_cache if args.text_cache and Path(args.text_cache).exists() else None,
        tritsp_hidden_size=args.tritsp_hidden_size,
        tritsp_cond_dim=args.tritsp_cond_dim,
        tritsp_num_heads=args.tritsp_num_heads,
        tritsp_condition_tokens=args.tritsp_condition_tokens,
        tritsp_tsp_condition_tokens=args.tritsp_tsp_condition_tokens,
        tritsp_spp_condition_tokens=args.tritsp_spp_condition_tokens,
        tritsp_condition_decoder_layers=args.tritsp_condition_decoder_layers,
        tritsp_delta_rank=args.tritsp_delta_rank,
        tritsp_gate_max=args.tritsp_gate_max,
        tritsp_inject_block_indices=args.tritsp_inject_block_indices,
        tritsp_high_boundary=args.high_boundary,
        tritsp_middle_boundary=args.middle_boundary,
    )


def cosine_schedule_lambda(step: int, total_steps: int, warmup_steps: int, min_ratio: float):
    if step < warmup_steps:
        return max(1e-4, float(step + 1) / max(1, warmup_steps))
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return float(min_ratio) + (1.0 - float(min_ratio)) * 0.5 * (1.0 + math.cos(math.pi * progress))


def to_float(x) -> float:
    return float(x.detach().float().cpu().item()) if torch.is_tensor(x) else float(x)


def diagnostics(parts: Mapping[str, torch.Tensor], total: torch.Tensor, reg: torch.Tensor) -> Dict[str, float]:
    out = {"total": to_float(total), "reg": to_float(reg)}
    for key in DIAG_KEYS_DUP_CMDC:
        if key in {"total", "reg"}:
            continue
        if key in parts:
            out[key] = to_float(parts[key])
    return out


def gather_objects(obj, world_size: int):
    if int(world_size) == 1:
        return [obj]
    values = [None for _ in range(int(world_size))]
    dist.all_gather_object(values, obj)
    return values


class FrozenPatchLPIPSDuPCmdc(torch.nn.Module):
    """Optional patch-LPIPS, used only for RGB 256x256-ish patches."""
    def __init__(self, device: str, net: str = "alex"):
        super().__init__()
        from lpips import LPIPS
        self.model = LPIPS(net=net).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.model(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean()


def fixed_random_indices(total: int, count: int, seed: int):
    total = int(total)
    count = max(0, min(int(count), total))
    rng = random.Random(int(seed))
    idx = list(range(total))
    rng.shuffle(idx)
    return idx[:count]


def parse_stage_cycle(text: str) -> List[str]:
    values = [x.strip().lower() for x in str(text).replace(";", ",").replace(" ", ",").split(",") if x.strip()]
    if not values or any(x not in {"high", "middle", "low"} for x in values):
        raise ValueError(f"--stage_cycle must contain only high,middle,low; got {text!r}")
    return values


def stage_for_pair_epoch(pair_key: str, epoch: int, cycle: Sequence[str]) -> str:
    """Deterministic per-pair stage schedule.

    With the default five-epoch cycle H/M/H/M/L, every pair sees exactly two High,
    two Middle and one Low stage in any complete five-epoch block. A stable pair-key
    offset prevents all images from using the same stage in the same epoch while
    preserving the global 0.4/0.4/0.2 exposure.
    """
    digest = hashlib.sha1(str(pair_key).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False) % len(cycle)
    return str(cycle[(int(epoch) + offset) % len(cycle)])


def load_modality(item, modality: str, args):
    rgb_path, thermal_path, _prompt_path, _codes, _pair_key = item
    if modality == "rgb":
        image, raw_hw, meta = load_rgb_tensor(
            rgb_path, args.preprocess_mode, args.target_width, args.target_height
        )
        return image, raw_hw, meta, str(rgb_path)
    image, _gray, raw_hw, meta = load_thermal_tensor(
        thermal_path, args.preprocess_mode, args.target_width, args.target_height
    )
    return image, raw_hw, meta, str(thermal_path)


def _crop_latent_patch_for_val(latent: torch.Tensor, crop_size: int, seed: int):
    h, w = int(latent.shape[-2]), int(latent.shape[-1])
    ch = min(max(1, int(crop_size)), h)
    cw = min(max(1, int(crop_size)), w)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    top = 0 if ch == h else int(torch.randint(0, h - ch + 1, (1,), generator=gen).item())
    left = 0 if cw == w else int(torch.randint(0, w - cw + 1, (1,), generator=gen).item())
    return latent[..., top:top + ch, left:left + cw]


def _gray3_val(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    g = x.mean(dim=1, keepdim=True)
    return g.repeat(1, 3, 1, 1)


def _psnr_val(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred.float() - target.float()) ** 2).detach().cpu().item()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))

def modality_backward(
    model,
    item,
    modality: str,
    edge: Tuple[int, int, int],
    negative_codes: Optional[Mapping[str, str]],
    config,
    args,
    noise_seed: int,
    loss_scale: float,
    use_patch_quality: bool = False,
    patch_lpips_model=None,
):
    image_cpu, raw_hw, meta, image_path = load_modality(item, modality, args)
    image = image_cpu.to(device=model.device, dtype=torch.float32, non_blocking=True)
    with torch.no_grad():
        z0 = model.image_to_latent(image).float()
    model.configure_empty(config.encoding_guidance_scale, image.shape[-1], image.shape[-2])
    current_t, next_t, _ = edge
    current_snr = model.get_timestep_snr(current_t).to(model.device)
    noisy, _ = make_noisy_latent(z0, current_snr, seed=int(noise_seed))
    correct_codes = item[3]


    parts = backward_losses_memory_efficient(
        model=model,
        z0=z0,
        noisy=noisy,
        current_t=current_t,
        next_t=next_t,
        correct_codes=correct_codes,
        negative_codes=negative_codes,
        modality=modality,
        num_pixels=int(raw_hw[0]) * int(raw_hw[1]),
        condition_mode=args.train_condition_mode,
        loss_scale=float(loss_scale),
        lambda_path=args.lambda_path,
        lambda_dist=args.lambda_dist,
        lambda_bridge=args.lambda_bridge,
        lambda_spec=args.lambda_spec,
        lambda_reg=args.lambda_reg,
        bridge_gain_reference=args.bridge_gain_reference,
        bridge_dist_margin_rel=args.bridge_dist_margin_rel,
        specificity_margin_rel=args.specificity_margin_rel,
        leak_allow_rel=args.leak_allow_rel,
        lambda_patch=(args.lambda_patch if use_patch_quality else 0.0),
        patch_crop_latent_size=args.patch_crop_latent_size,
        patch_seed=int(noise_seed) + 7919,
        patch_edge_weight=args.patch_edge_weight,
        patch_lpips_model=patch_lpips_model,
        patch_lpips_weight=args.patch_lpips_weight,
    )
    diag = {key: to_float(value) for key, value in parts.items() if key in DIAG_KEYS_DUP_CMDC}
    diag.update({
        "raw_height": int(raw_hw[0]),
        "raw_width": int(raw_hw[1]),
        "proc_height": int(image.shape[-2]),
        "proc_width": int(image.shape[-1]),
        "image_path": image_path,
        "prompt_path": str(item[2]),
        "pair_key": item[4],
        "thermal_conversion": str(meta.get("thermal_conversion", "not_applicable")),
    })
    del image_cpu, image, z0, noisy, parts
    return diag


@torch.no_grad()
def validate(model, val_items, val_edges, config, args, rank: int, world_size: int, epoch: int):
    """Lightweight all-pair validation without negative/counterfactual prompts."""
    model.tritsp_adapter.eval()
    local_modalities = ["rgb", "thermal"] if int(world_size) == 1 else (["rgb"] if int(rank) == 0 else ["thermal"])
    local_rows = []
    local_oom_skips = 0
    n = min(len(val_items), args.val_max_pairs) if int(args.val_max_pairs) > 0 else len(val_items)
    iterator = tqdm(range(n), desc=f"val-dup-cmdc e{epoch + 1} rank{rank}", disable=rank != 0)
    for item_idx in iterator:
        item = val_items[item_idx]
        for stage, edge in val_edges:
            for modality in local_modalities:
                try:
                    image_cpu, raw_hw, _meta, _ = load_modality(item, modality, args)
                    image = image_cpu.to(model.device, dtype=torch.float32)
                    z0 = model.image_to_latent(image).float()
                    model.configure_empty(config.encoding_guidance_scale, image.shape[-1], image.shape[-2])
                    current_snr = model.get_timestep_snr(edge[0]).to(model.device)
                    noisy, _ = make_noisy_latent(
                        z0, current_snr, args.val_noise_seed + item_idx * 101 + edge[2] + (0 if modality == "rgb" else 100000)
                    )
                    parts = compute_losses(
                        model=model, z0=z0, noisy=noisy, current_t=edge[0], next_t=edge[1],
                        correct_codes=item[3], negative_codes=None, modality=modality,
                        num_pixels=int(raw_hw[0]) * int(raw_hw[1]),
                        condition_mode=args.train_condition_mode,
                        bridge_gain_reference=args.bridge_gain_reference,
                        bridge_dist_margin_rel=args.bridge_dist_margin_rel,
                        specificity_margin_rel=args.specificity_margin_rel,
                        leak_allow_rel=args.leak_allow_rel,
                    )
                    row = {"stage": stage, "modality": modality}
                    for key in ("rho", "dist_ratio", "bridge_success", "kl_gain_rel"):
                        row[key] = to_float(parts[key])
                    local_rows.append(row)
                    del image_cpu, image, z0, noisy, parts
                except torch.cuda.OutOfMemoryError:
                    local_oom_skips += 1
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    gathered = gather_objects(local_rows, world_size)
    all_rows = [row for group in gathered for row in (group or [])]
    if int(world_size) > 1:
        oom_tensor = torch.tensor([local_oom_skips], device=model.device, dtype=torch.int64)
        dist.all_reduce(oom_tensor, op=dist.ReduceOp.SUM)
        total_oom_skips = int(oom_tensor.item())
    else:
        total_oom_skips = int(local_oom_skips)
    summary: Dict[str, float] = {
        "epoch": float(epoch + 1),
        "train_condition_mode": args.train_condition_mode,
        "val_pairs": float(n),
        "val_oom_skips": float(total_oom_skips),
    }
    macro_scores = []
    for stage in ("high", "middle", "low"):
        for modality in ("rgb", "thermal"):
            rows = [r for r in all_rows if r["stage"] == stage and r["modality"] == modality]
            if not rows:
                continue
            prefix = f"{stage}_{modality}"
            for key in ("rho", "dist_ratio", "bridge_success", "kl_gain_rel"):
                summary[f"{prefix}_{key}"] = sum(float(r[key]) for r in rows) / len(rows)
            
            macro_scores.append(summary[f"{prefix}_rho"] + 0.10 * summary[f"{prefix}_dist_ratio"])
    summary["macro_score"] = sum(macro_scores) / max(1, len(macro_scores))
    summary["all_stage_rho_below_1"] = float(all(
        summary.get(f"{stage}_{modality}_rho", 99.0) < 1.0
        for stage in ("high", "middle", "low") for modality in ("rgb", "thermal")
    ))
    model.tritsp_adapter.train()
    return summary


@torch.no_grad()
def patch_gain_validate(model, val_items, val_edges, config, args, rank: int, world_size: int, epoch: int, lpips_model=None):
    """All-pair (by default) H/M/L patch-gain validation for checkpoint selection.

    Base and Full use the same noisy edge. RGB and Thermal are aggregated separately
    so one modality cannot silently hide degradation in the other. Set
    --patch_val_pairs > 0 only for an explicit debugging subset; 0 means all pairs.
    """
    if int(args.patch_val_pairs) > 0:
        indices = fixed_random_indices(len(val_items), int(args.patch_val_pairs), int(args.patch_val_seed))
    else:
        indices = list(range(len(val_items)))
    local_modalities = ["rgb", "thermal"] if int(world_size) == 1 else (["rgb"] if int(rank) == 0 else ["thermal"])
    model.tritsp_adapter.eval()
    rows = []
    local_oom_skips = 0
    iterator = tqdm(indices, desc=f"patch-val-dup-cmdc e{epoch + 1} rank{rank}", disable=rank != 0)
    for subset_pos, item_idx in enumerate(iterator):
        item = val_items[int(item_idx)]
        for stage, edge in val_edges:
            for modality in local_modalities:
                image_cpu = None
                try:
                    image_cpu, raw_hw, _meta, image_path = load_modality(item, modality, args)
                    image = image_cpu.to(model.device, dtype=torch.float32)
                    z0 = model.image_to_latent(image).float()
                    model.configure_empty(config.encoding_guidance_scale, image.shape[-1], image.shape[-2])
                    current_snr = model.get_timestep_snr(edge[0]).to(model.device)
                    seed_offset = 0 if modality == "rgb" else 100000
                    noisy, _ = make_noisy_latent(
                        z0, current_snr, int(args.val_noise_seed) + int(item_idx) * 101 + int(edge[2]) + seed_offset
                    )
                    _kl_base, x0_base, _ = exact_edge_forward(
                        model, noisy, z0, edge[0], edge[1], None, modality, trainable=False
                    )
                    _kl_corr, x0_corr, _ = exact_edge_forward(
                        model, noisy, z0, edge[0], edge[1], item[3], modality, trainable=False,
                        condition_mode=args.train_condition_mode
                    )
                    crop_seed = int(args.patch_val_seed) + int(item_idx) * 1009 + int(edge[2]) * 17
                    z0_crop = _crop_latent_patch_for_val(z0, args.patch_val_crop_latent_size, crop_seed)
                    base_crop = _crop_latent_patch_for_val(x0_base, args.patch_val_crop_latent_size, crop_seed)
                    corr_crop = _crop_latent_patch_for_val(x0_corr, args.patch_val_crop_latent_size, crop_seed)
                    target_img = model.decode_latent_for_loss(z0_crop)
                    base_img = model.decode_latent_for_loss(base_crop)
                    corr_img = model.decode_latent_for_loss(corr_crop)
                    if modality == "thermal":
                        target_img = _gray3_val(target_img)
                        base_img = _gray3_val(base_img)
                        corr_img = _gray3_val(corr_img)
                    base_psnr = _psnr_val(base_img, target_img)
                    corr_psnr = _psnr_val(corr_img, target_img)
                    lpips_gain = float("nan")
                    if bool(args.patch_val_lpips) and lpips_model is not None and modality == "rgb":
                        base_lp = float(lpips_model(base_img, target_img).detach().cpu().item())
                        corr_lp = float(lpips_model(corr_img, target_img).detach().cpu().item())
                        lpips_gain = base_lp - corr_lp
                    rows.append({
                        "epoch": epoch + 1,
                        "train_condition_mode": args.train_condition_mode,
                        "subset_pos": subset_pos,
                        "item_index": int(item_idx),
                        "pair_key": item[4],
                        "stage": stage,
                        "modality": modality,
                        "base_patch_psnr": base_psnr,
                        "correct_patch_psnr": corr_psnr,
                        "patch_psnr_gain": corr_psnr - base_psnr,
                        "patch_lpips_gain": lpips_gain,
                        "image_path": str(image_path),
                        "prompt_path": str(item[2]),
                    })
                    del image, z0, noisy, x0_base, x0_corr, z0_crop, base_crop, corr_crop
                    del target_img, base_img, corr_img
                except torch.cuda.OutOfMemoryError:
                    local_oom_skips += 1
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                finally:
                    if image_cpu is not None:
                        del image_cpu
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    gathered = gather_objects(rows, world_size)
    all_rows = [r for group in gathered for r in (group or [])]
    if int(world_size) > 1:
        oom_tensor = torch.tensor([local_oom_skips], device=model.device, dtype=torch.int64)
        dist.all_reduce(oom_tensor, op=dist.ReduceOp.SUM)
        total_oom_skips = int(oom_tensor.item())
    else:
        total_oom_skips = int(local_oom_skips)
    if int(rank) == 0:
        manifest_path = Path(args.output_dir) / "patch_val_pairs_manifest.csv"
        if not manifest_path.exists():
            with open(manifest_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["subset_pos", "item_index", "pair_key", "rgb_path", "thermal_path", "prompt_path"])
                writer.writeheader()
                for subset_pos, item_idx in enumerate(indices):
                    it = val_items[int(item_idx)]
                    writer.writerow({"subset_pos": subset_pos, "item_index": item_idx, "pair_key": it[4], "rgb_path": str(it[0]), "thermal_path": str(it[1]), "prompt_path": str(it[2])})
        rows_path = Path(args.output_dir) / "patch_validation_rows.csv"
        for row in all_rows:
            append_csv(rows_path, row)

    summary = {
        "train_condition_mode": args.train_condition_mode,
        "patch_val_pairs": float(len(indices)),
        "patch_val_oom_skips": float(total_oom_skips),
    }
    gains = {}
    
    for stage in ("high", "middle", "low"):
        for modality in ("rgb", "thermal"):
            vals = [float(r["patch_psnr_gain"]) for r in all_rows if r.get("stage") == stage and r.get("modality") == modality and math.isfinite(float(r["patch_psnr_gain"]))]
            if vals:
                summary[f"patch_val_{stage}_{modality}_psnr_gain"] = sum(vals) / len(vals)
            lpvals = [float(r["patch_lpips_gain"]) for r in all_rows if r.get("stage") == stage and r.get("modality") == modality and math.isfinite(float(r.get("patch_lpips_gain", float("nan"))))]
            if lpvals:
                summary[f"patch_val_{stage}_{modality}_lpips_gain"] = sum(lpvals) / len(lpvals)

    for modality in ("rgb", "thermal"):
        vals = [float(r["patch_psnr_gain"]) for r in all_rows if r.get("modality") == modality and math.isfinite(float(r["patch_psnr_gain"]))]
        if vals:
            gains[modality] = sum(vals) / len(vals)
            summary[f"patch_val_{modality}_psnr_gain"] = gains[modality]
        lpvals = [float(r["patch_lpips_gain"]) for r in all_rows if r.get("modality") == modality and math.isfinite(float(r.get("patch_lpips_gain", float("nan"))))]
        if lpvals:
            summary[f"patch_val_{modality}_lpips_gain"] = sum(lpvals) / len(lpvals)

    rgb_gain = gains.get("rgb", 0.0)
    ir_gain = gains.get("thermal", 0.0)
    score = 0.5 * rgb_gain + 0.5 * ir_gain
    if rgb_gain < -0.02:
        score += 5.0 * rgb_gain
    if ir_gain < -0.02:
        score += 5.0 * ir_gain
    if "patch_val_rgb_lpips_gain" in summary:
        score += float(args.patch_val_lpips_weight) * float(summary["patch_val_rgb_lpips_gain"])
    summary["patch_val_score"] = score
    model.tritsp_adapter.train()
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="TriTSP-Bridge paired RGB-thermal adapter training")
    p.add_argument("--config", required=True)
    p.add_argument("--rgb_list", required=True)
    p.add_argument("--thermal_list", "--ir_list", dest="thermal_list", required=True)
    p.add_argument("--prompt_list", required=True, help="TSP JSON list; aligned to RGB/thermal by last-folder/image-basename")
    p.add_argument("--val_rgb_list", required=True)
    p.add_argument("--val_thermal_list", "--val_ir_list", dest="val_thermal_list", required=True)
    p.add_argument("--val_prompt_list", required=True, help="Validation TSP JSON list; key-aligned, not line-aligned")
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--text_cache", default=None)
    p.add_argument("--resume_ckpt", default=None)

    p.add_argument("--preprocess_mode", default="pad64", choices=["resize", "pad64", "none"])
    p.add_argument("--target_width", type=int, default=1024)
    p.add_argument("--target_height", type=int, default=576)

    p.add_argument("--stop_timestep", type=int, default=45)
    p.add_argument("--high_boundary", type=int, default=300)
    p.add_argument("--middle_boundary", type=int, default=85)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr_schedule_epochs", type=int, default=40, help="Cosine LR horizon, separated from --epochs so a 24-epoch run can resume later without restarting LR")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--save_every", type=int, default=2)
    p.add_argument("--val_every", type=int, default=2)
    p.add_argument("--val_max_pairs", type=int, default=0, help="0 = all matched validation pairs")
    p.add_argument("--val_edges_per_stage", type=int, default=1)
    p.add_argument("--val_noise_seed", type=int, default=73117)

    p.add_argument("--lambda_path", type=float, default=1.0)
    p.add_argument("--lambda_dist", type=float, default=0.10)
    p.add_argument("--lambda_bridge", type=float, default=0.20)
    p.add_argument("--lambda_spec", type=float, default=0.0, help="Detached semantic-negative auxiliary weight; default 0 for v7 quality-first")
    p.add_argument("--lambda_patch", type=float, default=0.03, help="Light decoded-patch quality loss weight")
    p.add_argument("--lambda_reg", type=float, default=1e-4)
    p.add_argument("--bridge_gain_reference", type=float, default=0.02)
    p.add_argument("--bridge_dist_margin_rel", type=float, default=0.002)
    p.add_argument("--specificity_margin_rel", type=float, default=0.005)
    p.add_argument("--leak_allow_rel", type=float, default=0.005)
    p.add_argument("--stage_cycle", default="high,middle,high,middle,low", help="Deterministic per-pair cycle; default gives 0.4/0.4/0.2 exposure in every 5 epochs")
    p.add_argument("--semantic_negative_prob", type=float, default=0.0, help="Probability of detached semantic-opposite negative; keep 0 for first v7 run")
    p.add_argument("--semantic_negative_start_epoch", type=int, default=20)
    p.add_argument("--patch_quality_probability", type=float, default=0.10)
    p.add_argument("--patch_crop_latent_size", type=int, default=32, help="32 latent ~= 256x256 image patch for SANA VAE scale 8")
    p.add_argument("--patch_edge_weight", type=float, default=0.20)
    p.add_argument("--patch_lpips_weight", type=float, default=0.05)
    p.add_argument("--use_patch_lpips", action="store_true")
    p.add_argument("--patch_val_pairs", type=int, default=0, help="0 = all matched validation pairs; positive value = fixed-random debug subset")
    p.add_argument("--patch_val_seed", type=int, default=20260714)
    p.add_argument("--patch_val_every", type=int, default=4)
    p.add_argument("--patch_val_crop_latent_size", type=int, default=32)
    p.add_argument("--patch_val_lpips", action="store_true")
    p.add_argument("--patch_val_lpips_weight", type=float, default=0.50)

    p.add_argument("--tritsp_hidden_size", type=int, default=None)
    p.add_argument("--tritsp_cond_dim", type=int, default=512)
    p.add_argument("--tritsp_num_heads", type=int, default=8)
    p.add_argument("--tritsp_condition_tokens", type=int, default=12, help="Total query budget; must equal TSP+SPP query counts")
    p.add_argument("--tritsp_tsp_condition_tokens", type=int, default=8, help="Independent TSP query bank size")
    p.add_argument("--tritsp_spp_condition_tokens", type=int, default=4, help="Independent SPP query bank size")
    p.add_argument("--tritsp_condition_decoder_layers", type=int, default=2)
    p.add_argument("--tritsp_delta_rank", type=int, default=64)
    p.add_argument("--tritsp_gate_max", type=float, default=0.5)
    p.add_argument("--tritsp_inject_block_indices", default="9,14,18,23")
    p.add_argument("--lowrank_warmup_epochs", type=int, default=4, help="Freeze LowRank Delta for the initial N epochs")
    p.add_argument("--branch_log_every", type=int, default=200, help="Record branch contribution ratios every N optimizer steps; 0 disables training-time logger")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--keep_text_encoder_on_gpu", action="store_true")
    args = p.parse_args()
    args.train_condition_mode = "full"
    return args


def main():
    args = parse_args()
    if int(args.epochs) <= 0 or int(args.lr_schedule_epochs) <= 0:
        raise ValueError("--epochs and --lr_schedule_epochs must be positive")
    if int(args.tritsp_tsp_condition_tokens) + int(args.tritsp_spp_condition_tokens) != int(args.tritsp_condition_tokens):
        raise ValueError(
            "--tritsp_tsp_condition_tokens + --tritsp_spp_condition_tokens must equal --tritsp_condition_tokens"
        )
    if int(args.lowrank_warmup_epochs) < 0:
        raise ValueError("--lowrank_warmup_epochs must be >= 0")
    if int(args.branch_log_every) < 0:
        raise ValueError("--branch_log_every must be >= 0")
    if int(args.val_every) <= 0:
        raise ValueError("--val_every must be positive")
    if int(args.patch_val_every) > 0 and int(args.patch_val_every) % int(args.val_every) != 0:
        raise ValueError("--patch_val_every must be a multiple of --val_every so patch rows share the same validation summary")
    distributed, rank, local_rank, world_size = init_distributed()
    if world_size not in (1, 2):
        raise ValueError("TriTSP training supports one GPU or exactly two GPUs")
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    event_log = output / "events.log"
    if rank == 0:
        with open(output / "args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)

    config = load_config(args.config)
    train_items = build_paired_items(args.rgb_list, args.thermal_list, args.prompt_list)
    val_items = build_paired_items(args.val_rgb_list, args.val_thermal_list, args.val_prompt_list)
    if rank == 0:
        write_pair_manifest(train_items, output / "train_pair_manifest.csv")
        write_pair_manifest(val_items, output / "val_pair_manifest.csv")
    all_codes = [item[3] for item in train_items + val_items]
    edges = build_path_edges_to_stop(config.encoding_timesteps, args.stop_timestep)
    stage_edges = split_edges_by_stage(edges, args.high_boundary, args.middle_boundary)
    val_edges = representative_edges(stage_edges, args.val_edges_per_stage)

    model = make_model(config, args, device)
    cache_path = Path(args.text_cache) if args.text_cache else output / "text_cache.pt"
    
    
    if rank == 0:
        if cache_path.exists():
            try:
                model.tritsp_text_cache = TriTSPTextCache.load(cache_path)
            except ValueError:
                model.tritsp_text_cache = TriTSPTextCache(num_stage_tokens=8, num_field_tokens=2)
        log_event(event_log, f"Validating/building frozen text cache -> {cache_path}")
        model.prebuild_text_cache(all_codes, cache_path)
    barrier()
    model.tritsp_text_cache = TriTSPTextCache.load(cache_path)
    model.prepare_base_prompt_cache([config.encoding_guidance_scale, config.denoising_guidance_scale])
    if not args.keep_text_encoder_on_gpu:
        model.offload_text_encoder()
    if args.gradient_checkpointing:
        if rank == 0:
            log_event(event_log, "[WARN] --gradient_checkpointing is ignored in dup-cmdc to avoid dynamic prompt recompute mismatch")

    patch_lpips_model = None
    if args.use_patch_lpips or args.patch_val_lpips:
        patch_lpips_model = FrozenPatchLPIPSDuPCmdc(device)
        if rank == 0:
            log_event(event_log, "Enabled RGB decoded-patch LPIPS supervision / validation")

    params = model.trainable_parameters(args.train_condition_mode)
    broadcast_parameters(params, src=0)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    schedule_steps = max(1, int(args.lr_schedule_epochs) * len(train_items))
    min_ratio = args.min_lr / max(args.lr, 1e-12)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule_lambda(step, schedule_steps, args.warmup_steps, min_ratio),
    )
    start_epoch = 0
    global_step = 0
    best_macro_score = float("inf")
    best_patch_score = -float("inf")
    if args.resume_ckpt:
        resume_probe = torch.load(Path(args.resume_ckpt), map_location="cpu", weights_only=False)
        resume_extra = resume_probe.get("extra") or {}
        if str(resume_extra.get("train_condition_mode", (resume_extra.get("args") or {}).get("train_condition_mode", "full"))).lower() != "full":
            raise ValueError("Cannot resume a non-full checkpoint")
        del resume_probe
        extra = model.load_checkpoint(args.resume_ckpt, optimizer=optimizer, scheduler=scheduler)
        start_epoch = int(extra.get("epoch", 0))
        global_step = int(extra.get("global_step", 0))
        best_macro_score = float(extra.get("best_macro_score", extra.get("best_score", float("inf"))))
        best_patch_score = float(extra.get("best_patch_score", -float("inf")))
        saved_horizon = extra.get("lr_schedule_epochs", None)
        if saved_horizon is not None and int(saved_horizon) != int(args.lr_schedule_epochs) and rank == 0:
            log_event(
                event_log,
                f"[WARN] Resume LR horizon changed from {saved_horizon} to {args.lr_schedule_epochs}. "
                "For exact continuation, keep --lr_schedule_epochs unchanged.",
            )
        saved_lowrank_warmup = extra.get(
            "lowrank_warmup_epochs",
            extra.get("args", {}).get("lowrank_warmup_epochs", None),
        )
        if saved_lowrank_warmup is not None and int(saved_lowrank_warmup) != int(args.lowrank_warmup_epochs):
            raise ValueError(
                f"Resume warm-up mismatch: checkpoint lowrank_warmup_epochs={saved_lowrank_warmup}, "
                f"current --lowrank_warmup_epochs={args.lowrank_warmup_epochs}. "
                "Keep the condition-first schedule unchanged for an exact continuation."
            )
        if start_epoch >= int(args.epochs):
            raise ValueError(f"Resume checkpoint is already at epoch {start_epoch}, but --epochs={args.epochs}. Set --epochs to a larger total target.")

    if rank == 0:
        log_event(event_log, f"world_size={world_size}, train_pairs={len(train_items)}, val_pairs={len(val_items)}, target_epochs={args.epochs}, lr_schedule_epochs={args.lr_schedule_epochs}")
        if len(train_items) != 1080:
            log_event(event_log, "[WARN] Expected full training split is 1080 pairs; using the actually matched training pairs.")
        log_event(event_log, f"stage edge counts: high={len(stage_edges['high'])}, middle={len(stage_edges['middle'])}, low={len(stage_edges['low'])}")
        log_event(event_log, f"full training: mode={args.train_condition_mode}; stage_cycle={args.stage_cycle}; lambda_patch={args.lambda_patch}; semantic_negative_prob={args.semantic_negative_prob}; negative validation disabled")
        log_event(
            event_log,
            f"split-prior DPMA: TSP->CrossAttn ({args.tritsp_tsp_condition_tokens} queries), "
            f"SPP->FiLM ({args.tritsp_spp_condition_tokens} queries), LowRank warmup_epochs={args.lowrank_warmup_epochs}; "
            f"branch_log_every={args.branch_log_every}",
        )

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            lowrank_active = int(epoch) >= int(args.lowrank_warmup_epochs)
            model.set_lowrank_trainable(lowrank_active)
            if rank == 0:
                log_event(event_log, f"epoch {epoch + 1}: lowrank_delta_trainable={int(lowrank_active)}")
            order = list(range(len(train_items)))
            epoch_rng = random.Random(args.seed + epoch)
            epoch_rng.shuffle(order)
            
            
            stage_cycle = parse_stage_cycle(args.stage_cycle)
            epoch_rows = []
            stage_counts = defaultdict(int)
            negative_counts = defaultdict(int)
            oom_skips = 0
            iterator = tqdm(order, desc=f"train-dup-cmdc e{epoch + 1}/{args.epochs}", disable=rank != 0)
            for local_step, item_idx in enumerate(iterator):
                item = train_items[item_idx]
                decision_rng = random.Random(args.seed + epoch * 1000003 + item_idx * 97)
                stage = stage_for_pair_epoch(item[4], epoch, stage_cycle)
                edge = decision_rng.choice(stage_edges[stage])
                
                
                use_negative = (
                    epoch >= int(args.semantic_negative_start_epoch)
                    and decision_rng.random() < float(args.semantic_negative_prob)
                    and float(args.lambda_spec) > 0.0
                )
                neg_mode = "semantic_far" if use_negative else "none"
                negative = None if not use_negative else generate_counterfactual(
                    item[3], "far", args.seed + epoch * 2000003 + item_idx * 193
                )
                neg_distance = 0.0 if negative is None else semantic_distance(item[3], negative)
                do_branch_log = (
                    int(args.branch_log_every) > 0
                    and int(global_step) % int(args.branch_log_every) == 0
                )
                model.reset_branch_contribution_stats()
                model.set_branch_logging(do_branch_log)
                optimizer.zero_grad(set_to_none=True)
                local_diags = []
                oom_flag = 0
                try:
                    if world_size == 1:
                        modalities = ["rgb", "thermal"]
                        scale = 0.5
                    else:
                        modalities = ["rgb"] if rank == 0 else ["thermal"]
                        scale = 1.0
                    for modality in modalities:
                        seed_offset = 0 if modality == "rgb" else 100000000
                        use_patch_quality = decision_rng.random() < float(args.patch_quality_probability)
                        diag = modality_backward(
                            model, item, modality, edge, negative, config, args,
                            noise_seed=args.seed + global_step * 17 + edge[2] + seed_offset,
                            loss_scale=scale,
                            use_patch_quality=use_patch_quality,
                            patch_lpips_model=patch_lpips_model,
                        )
                        diag["patch_quality_step"] = int(use_patch_quality)
                        diag["modality"] = modality
                        local_diags.append(diag)
                except torch.cuda.OutOfMemoryError as exc:
                    oom_flag = 1
                    optimizer.zero_grad(set_to_none=True)
                    local_diags = []
                    if rank == 0:
                        log_event(event_log, f"OOM at epoch={epoch + 1}, pair={item_idx}, stage={stage}, edge={edge}: {exc}")
                if distributed:
                    flag_tensor = torch.tensor([oom_flag], device=device, dtype=torch.int32)
                    dist.all_reduce(flag_tensor, op=dist.ReduceOp.MAX)
                    oom_flag = int(flag_tensor.item())
                if oom_flag:
                    oom_skips += 1
                    optimizer.zero_grad(set_to_none=True)
                    model.set_branch_logging(False)
                    model.reset_branch_contribution_stats()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                average_gradients(params, world_size)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip).detach().cpu())
                optimizer.step()
                scheduler.step()
                global_step += 1

                model.set_branch_logging(False)
                local_branch_rows = (
                    model.consume_branch_contribution_stats(reset=True)
                    if do_branch_log else []
                )
                gathered_branch_rows = gather_objects(local_branch_rows, world_size)
                if rank == 0 and do_branch_log:
                    for branch_group in gathered_branch_rows:
                        for branch_row in (branch_group or []):
                            branch_row = dict(branch_row)
                            slot = int(branch_row.get("slot", -1))
                            branch_row.update({
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "pair_index": item_idx,
                                "pair_key": item[4],
                                "current_t": edge[0],
                                "next_t": edge[1],
                                "lowrank_trainable": int(lowrank_active),
                                "block_index": (
                                    int(model.tritsp_inject_block_indices[slot])
                                    if 0 <= slot < len(model.tritsp_inject_block_indices) else -1
                                ),
                            })
                            append_csv(
                                output / "branch_contributions_train.csv", branch_row
                            )

                gathered_diags = gather_objects(local_diags, world_size)
                merged_diags = [d for group in gathered_diags for d in (group or [])]
                row = {
                    "epoch": epoch + 1,
                    "train_condition_mode": args.train_condition_mode,
                    "global_step": global_step,
                    "pair_index": item_idx,
                    "pair_key": item[4],
                    "rgb_path": str(item[0]),
                    "thermal_path": str(item[1]),
                    "prompt_path": str(item[2]),
                    "stage": stage,
                    "current_t": edge[0],
                    "next_t": edge[1],
                    "edge_idx": edge[2],
                    "negative_mode": neg_mode,
                    "negative_semantic_distance": neg_distance,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm,
                    "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0,
                    "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0.0,
                    "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0,
                }
                for diag in merged_diags:
                    mod = diag.pop("modality")
                    for key, value in diag.items():
                        row[f"{mod}_{key}"] = value
                if rank == 0:
                    append_jsonl(output / "train_steps.jsonl", row)
                epoch_rows.append(row)
                stage_counts[stage] += 1
                negative_counts[neg_mode] += 1
                if rank == 0:
                    display_rho = row.get("rgb_rho", row.get("thermal_rho", float("nan")))
                    iterator.set_postfix(stage=stage, rho=f"{display_rho:.4f}", neg=neg_mode, lr=f"{row['lr']:.2e}")

                del local_diags, gathered_diags, merged_diags
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            
            if rank == 0:
                epoch_summary = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "seconds": time.time() - epoch_start,
                    "steps": len(epoch_rows),
                    "oom_skips": oom_skips,
                    "lr": optimizer.param_groups[0]["lr"],
                    "lowrank_trainable": int(lowrank_active),
                    "high_steps": stage_counts["high"],
                    "middle_steps": stage_counts["middle"],
                    "low_steps": stage_counts["low"],
                    "negative_none": negative_counts["none"],
                    "negative_far": negative_counts["far"],
                    "negative_hard": negative_counts["hard"],
                    "negative_semantic_far": negative_counts["semantic_far"],
                }
                numeric_keys = sorted({k for r in epoch_rows for k, v in r.items() if isinstance(v, (int, float)) and (k.startswith("rgb_") or k.startswith("thermal_"))})
                for key in numeric_keys:
                    vals = [float(r[key]) for r in epoch_rows if key in r and math.isfinite(float(r[key]))]
                    if vals:
                        epoch_summary[f"mean_{key}"] = sum(vals) / len(vals)
                append_csv(output / "train_epochs.csv", epoch_summary)
                log_event(event_log, f"epoch {epoch + 1} complete: steps={len(epoch_rows)}, oom_skips={oom_skips}, seconds={epoch_summary['seconds']:.1f}")


            
            extra = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_condition_mode": args.train_condition_mode,
                "best_macro_score": best_macro_score,
                "best_patch_score": best_patch_score,
                "lr_schedule_epochs": int(args.lr_schedule_epochs),
                "lowrank_warmup_epochs": int(args.lowrank_warmup_epochs),
                "args": vars(args),
            }
            if rank == 0:
                model.save_checkpoint(output / "latest.pt", extra=extra, optimizer=optimizer, scheduler=scheduler)
            barrier()

            val_summary = None
            if (epoch + 1) % args.val_every == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                val_summary = validate(model, val_items, val_edges, config, args, rank, world_size, epoch)
                if int(args.patch_val_every) > 0 and (epoch + 1) % int(args.patch_val_every) == 0:
                    patch_summary = patch_gain_validate(
                        model, val_items, val_edges, config, args, rank, world_size, epoch, lpips_model=patch_lpips_model
                    )
                    val_summary.update(patch_summary)
                if rank == 0:
                    append_csv(output / "validation_epochs.csv", val_summary)
                    log_event(
                        event_log,
                        f"validation epoch={epoch + 1}, macro_score={val_summary['macro_score']:.6f}, "
                        f"patch_val_score={val_summary.get('patch_val_score', float('nan')):.6f}, "
                        f"all_stage_rho_below_1={val_summary['all_stage_rho_below_1']}"
                    )
                    macro_score = float(val_summary["macro_score"])
                    if macro_score < best_macro_score:
                        best_macro_score = macro_score
                        best_extra = {
                            "epoch": epoch + 1, "global_step": global_step,
                            "train_condition_mode": args.train_condition_mode,
                            "best_macro_score": best_macro_score, "best_patch_score": best_patch_score,
                            "lr_schedule_epochs": int(args.lr_schedule_epochs), "lowrank_warmup_epochs": int(args.lowrank_warmup_epochs), "args": vars(args),
                            "val_summary": val_summary,
                        }
                        model.save_checkpoint(
                            output / "best_val_macro.pt", extra=best_extra, optimizer=optimizer, scheduler=scheduler
                        )
                    if "patch_val_score" in val_summary:
                        patch_score = float(val_summary["patch_val_score"])
                        if patch_score > best_patch_score:
                            best_patch_score = patch_score
                            best_extra = {
                                "epoch": epoch + 1, "global_step": global_step,
                                "train_condition_mode": args.train_condition_mode,
                                "best_macro_score": best_macro_score, "best_patch_score": best_patch_score,
                                "lr_schedule_epochs": int(args.lr_schedule_epochs), "lowrank_warmup_epochs": int(args.lowrank_warmup_epochs), "args": vars(args),
                                "val_summary": val_summary,
                            }
                            model.save_checkpoint(
                                output / "best_val_patch_gain.pt", extra=best_extra, optimizer=optimizer, scheduler=scheduler
                            )
                barrier()


            extra = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_condition_mode": args.train_condition_mode,
                "best_macro_score": best_macro_score,
                "best_patch_score": best_patch_score,
                "lr_schedule_epochs": int(args.lr_schedule_epochs),
                "lowrank_warmup_epochs": int(args.lowrank_warmup_epochs),
                "args": vars(args),
            }
            if val_summary is not None:
                extra["val_summary"] = val_summary
            if rank == 0:
                model.save_checkpoint(output / "latest.pt", extra=extra, optimizer=optimizer, scheduler=scheduler)
                if (epoch + 1) % args.save_every == 0:
                    model.save_checkpoint(
                        output / f"epoch_{epoch + 1:04d}.pt", extra=extra, optimizer=optimizer, scheduler=scheduler
                    )
            barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
