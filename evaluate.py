# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
import yaml
from easydict import EasyDict as edict
from PIL import Image

from lib import image_utils, metrics
from lib.diffc.encode import encode
from lib.diffc.rcc.gaussian_channel_simulator import GaussianChannelSimulator
from lib.models.SanaDiT_dup_cmdc import SanaDiTModel_DuPCmdc
from lib.dup_cmdc.distributed import (
    barrier,
    cleanup_distributed,
    init_distributed,
)
from lib.dup_cmdc.image_io import (
    crop_to_raw,
    gray3,
    load_rgb_tensor,
    load_thermal_tensor,
)
from lib.dup_cmdc.pairing import (
    PairedItemTriTSP,
    build_paired_items,
    write_pair_manifest,
)
from lib.dup_cmdc.text_cache import TriTSPTextCache
from compress import pack_tsp


SUPPORTED_CONDITION_MODES_DUP_CMDC = ("full",)


def normalize_condition_mode(mode: str) -> str:
    if str(mode).strip().lower() != "full":
        raise ValueError("Only full mode is supported")
    return "full"


def condition_flags(mode: str) -> tuple[bool, bool]:
    normalize_condition_mode(mode)
    return True, True


def checkpoint_architecture(path: str | Path) -> str | None:
    try:
        ckpt = torch.load(Path(path), map_location="cpu")
        if not isinstance(ckpt, dict):
            return None
        value = ckpt.get("architecture_version", None)
        return None if value is None else str(value)
    except Exception:
        return None


def checkpoint_train_mode(path: str | Path) -> str | None:
    """Read the formal training mode from checkpoint metadata without building SANA."""
    try:
        ckpt = torch.load(Path(path), map_location="cpu")
        extra = ckpt.get("extra", {}) if isinstance(ckpt, dict) else {}
        mode = extra.get("train_condition_mode")
        if mode is None and isinstance(extra.get("args"), dict):
            mode = extra["args"].get("train_condition_mode")
        if mode is None:
            return None
        return normalize_condition_mode(str(mode))
    except Exception:
        return None


def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return edict(yaml.safe_load(f))


def parse_timesteps(text: str, schedule: Sequence[int]) -> List[int]:
    values = [
        int(x)
        for x in str(text).replace(";", ",").replace(" ", ",").split(",")
        if x.strip()
    ]
    if not values:
        raise ValueError("--recon_timesteps must contain at least one timestep")
    missing = [t for t in values if t not in schedule]
    if missing:
        raise ValueError(f"Recon timesteps not in config.encoding_timesteps: {missing}")
    return sorted(set(values), reverse=True)


def truncate_config(config, recon_timesteps: Sequence[int]):
    cfg = edict(copy.deepcopy(dict(config)))
    schedule = [int(t) for t in cfg.encoding_timesteps]
    stop_t = min(int(t) for t in recon_timesteps)
    stop_idx = schedule.index(stop_t)
    old_len = len(schedule)
    cfg.encoding_timesteps = schedule[: stop_idx + 1]
    if getattr(cfg, "manual_dkl_per_step", None) is not None:
        manual = list(cfg.manual_dkl_per_step)
        if len(manual) != old_len:
            raise ValueError(
                f"manual_dkl_per_step length={len(manual)} != encoding_timesteps length={old_len}"
            )
        cfg.manual_dkl_per_step = manual[: stop_idx + 1]
    cfg.recon_timesteps = list(recon_timesteps)
    return cfg


def get_root(paths: Sequence[Path], explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if len(paths) == 1:
        return paths[0].parent
    return Path(os.path.commonpath([str(p.parent) for p in paths])).resolve()


def rel_path(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        
        return Path(path.parent.name) / path.name


def save_thermal_gray(tensor3: torch.Tensor, path: Path) -> None:
    gray = tensor3.mean(dim=1, keepdim=True).clamp(0, 1)
    arr = (
        gray[0, 0].detach().cpu().numpy() * 255.0
    ).round().clip(0, 255).astype("uint8")
    Image.fromarray(arr, mode="L").save(path)


def make_model(config, args, device: str):
    return SanaDiTModel_DuPCmdc(
        model_id=config.get("model_path", config.get("model_id")),
        device=device,
        transformer_dtype=config.get("transformer_dtype", "float16"),
        text_vae_dtype=config.get("text_vae_dtype", "bfloat16"),
        local_files_only=config.get("local_files_only", True),
        max_sequence_length=config.get("max_sequence_length", 300),
        tritsp_ckpt=args.checkpoint,
        tritsp_text_cache_path=(
            args.text_cache
            if args.text_cache and Path(args.text_cache).exists()
            else None
        ),
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


def _existing_result_keys(csv_path: Path) -> set[tuple[str, int]]:
    if not csv_path.exists():
        return set()
    try:
        frame = pd.read_csv(csv_path)
        if not {"pair_key", "recon_timestep"}.issubset(frame.columns):
            return set()
        return {
            (str(row.pair_key), int(row.recon_timestep))
            for row in frame[["pair_key", "recon_timestep"]].itertuples(index=False)
        }
    except Exception:
        return set()


def _load_existing_rows(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        return []
    try:
        return pd.read_csv(csv_path).to_dict("records")
    except Exception:
        return []


def evaluate_modality(
    model,
    modality: str,
    items: Sequence[PairedItemTriTSP],
    image_root: Path,
    output_root: Path,
    condition_mode: str,
    config,
    args,
    simulator,
    rank: int,
):
    use_stage_prompt, use_tsp_prompt = condition_flags(condition_mode)
    mode_root = output_root / condition_mode
    result_csv = mode_root / f"results_{modality}.csv"
    results: List[Dict] = _load_existing_rows(result_csv) if args.resume else []
    completed = _existing_result_keys(result_csv) if args.resume else set()
    branch_csv = mode_root / f"branch_contributions_{modality}.csv"
    branch_results: List[Dict] = (
        _load_existing_rows(branch_csv)
        if args.resume and bool(args.branch_contribution_log) else []
    )
    branch_completed_pairs = {
        str(row.get("pair_key")) for row in branch_results if row.get("pair_key") is not None
    }
    modality_dir = mode_root / modality
    modality_dir.mkdir(parents=True, exist_ok=True)

    iterator = range(len(items))
    for pair_idx in iterator:
        rgb_path, thermal_path, prompt_path, codes, pair_key = items[pair_idx]
        image_path = rgb_path if modality == "rgb" else thermal_path
        pending_timesteps = [
            int(t)
            for t in config.recon_timesteps
            if (pair_key, int(t)) not in completed
        ]
        if not pending_timesteps:
            continue

        if modality == "rgb":
            gt_cpu, raw_hw, meta = load_rgb_tensor(
                image_path,
                args.preprocess_mode,
                args.target_width,
                args.target_height,
            )
        else:
            gt_cpu, _gray, raw_hw, meta = load_thermal_tensor(
                image_path,
                args.preprocess_mode,
                args.target_width,
                args.target_height,
            )

        raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
        proc_h, proc_w = int(gt_cpu.shape[-2]), int(gt_cpu.shape[-1])
        rel = rel_path(image_path, image_root)
        do_branch_log = bool(args.branch_contribution_log) and pair_key not in branch_completed_pairs
        model.reset_branch_contribution_stats()
        model.set_branch_logging(do_branch_log)
        try:
            with torch.inference_mode():
                gt = gt_cpu.to(model.device, dtype=torch.float32)
                gt_latent = model.image_to_latent(gt)

                model.set_condition(
                    codes if use_tsp_prompt else None,
                    modality,
                    enabled=True,
                    condition_mode=condition_mode,
                )
                model.configure_empty(
                    config.encoding_guidance_scale, proc_w, proc_h
                )
                seeds_per_step, dkl_per_step, noisy_recons, step_indices = encode(
                    gt_latent,
                    config.encoding_timesteps,
                    model,
                    simulator,
                    config.manual_dkl_per_step,
                    config.recon_timesteps,
                    seed=args.compression_seed + pair_idx,
                )

                model.configure_empty(
                    config.denoising_guidance_scale, proc_w, proc_h
                )
                model.set_condition(
                    codes if use_tsp_prompt else None,
                    modality,
                    enabled=True,
                    condition_mode=condition_mode,
                )

                for noisy_recon, step_idx in zip(noisy_recons, step_indices):
                    timestep = int(config.encoding_timesteps[step_idx])
                    if (pair_key, timestep) in completed:
                        continue

                    byte_stream = simulator.compress_chunk_seeds(
                        seeds_per_step[: step_idx + 1],
                        dkl_per_step[: step_idx + 1],
                    )
                    image_bits = int(len(byte_stream) * 8)
                    prompt_bits_pair = len(pack_tsp(codes)) * 8 if use_tsp_prompt else 0
                    raw_pixels = max(1, raw_h * raw_w)
                    proc_pixels = max(1, proc_h * proc_w)

                    recon_latent = model.denoise(
                        noisy_recon, timestep, config.denoising_timesteps
                    )
                    if not torch.isfinite(recon_latent).all():
                        raise FloatingPointError(
                            f"recon_latent contains NaN/Inf: pair={pair_key}, "
                            f"modality={modality}, timestep={timestep}"
                        )
                    recon = model.latent_to_image(recon_latent)
                    if not torch.isfinite(recon).all():
                        raise FloatingPointError(
                            f"reconstruction contains NaN/Inf: pair={pair_key}, "
                            f"modality={modality}, timestep={timestep}"
                        )

                    gt_metric = crop_to_raw(gt, raw_hw)
                    recon_metric = crop_to_raw(recon, raw_hw)
                    if modality == "thermal":
                        gt_metric_for_metrics = gray3(gt_metric)
                        recon_metric_for_metrics = gray3(recon_metric)
                    else:
                        gt_metric_for_metrics = gt_metric
                        recon_metric_for_metrics = recon_metric

                    psnr = metrics.get_psnr(
                        recon_metric_for_metrics, gt_metric_for_metrics
                    )
                    lpips_value = (
                        float("nan")
                        if args.skip_lpips
                        else metrics.get_lpips(
                            recon_metric_for_metrics, gt_metric_for_metrics
                        )
                    )

                    recon_path = modality_dir / str(timestep).zfill(3) / rel
                    recon_path.parent.mkdir(parents=True, exist_ok=True)
                    if modality == "thermal":
                        save_thermal_gray(
                            recon_metric_for_metrics, recon_path
                        )
                    else:
                        image_utils.torch_to_pil_img(
                            recon_metric_for_metrics
                        ).save(recon_path)

                    prefix = "rgb" if modality == "rgb" else "thermal"
                    content = codes.get("content", {})
                    human = content.get("human", {})
                    vehicle = content.get("vehicle", {})
                    row = {
                        "pair_index": pair_idx,
                        "pair_key": pair_key,
                        "condition_mode": condition_mode,
                        "modality": modality,
                        "gt_path": str(image_path),
                        "rgb_path": str(rgb_path),
                        "thermal_path": str(thermal_path),
                        "prompt_path": str(prompt_path),
                        "recon_path": str(recon_path),
                        "recon_step_idx": int(step_idx),
                        "recon_timestep": timestep,
                        "raw_width": raw_w,
                        "raw_height": raw_h,
                        "proc_width": proc_w,
                        "proc_height": proc_h,
                        "preprocess_mode": args.preprocess_mode,
                        "thermal_conversion": str(
                            meta.get("thermal_conversion", "not_applicable")
                        ),
                        f"{prefix}_image_bits": image_bits,
                        f"{prefix}_image_bpp_proc": image_bits / proc_pixels,
                        f"{prefix}_image_bpp_raw": image_bits / raw_pixels,
                        "prompt_bits_pair_shared": prompt_bits_pair,
                        "prompt_bpp_proc_shared": prompt_bits_pair / proc_pixels,
                        "prompt_bpp_raw_shared": prompt_bits_pair / raw_pixels,
                        "stage_prompt_bits": 0,
                        "stage_prompt_bpp": 0.0,
                        "stage_prompt_used": int(use_stage_prompt),
                        "tsp_prompt_used": int(use_tsp_prompt),
                        "psnr": float(psnr),
                        "lpips": float(lpips_value),
                        "tsp_geometry": codes.get("geometry", ""),
                        "tsp_environment": ";".join(codes.get("environment", [])),
                        "tsp_human_present": int(bool(human.get("present", False))),
                        "tsp_human_organization": human.get("organization", ""),
                        "tsp_human_regions": ";".join(human.get("regions", [])),
                        "tsp_human_topology": human.get("topology", ""),
                        "tsp_human_morphology": ";".join(human.get("morphology", [])),
                        "tsp_vehicle_present": int(bool(vehicle.get("present", False))),
                        "tsp_vehicle_organization": vehicle.get("organization", ""),
                        "tsp_vehicle_regions": ";".join(vehicle.get("regions", [])),
                        "tsp_vehicle_topology": vehicle.get("topology", ""),
                        "tsp_vehicle_morphology": ";".join(vehicle.get("morphology", [])),
                    }
                    results.append(row)
                    completed.add((pair_key, timestep))
                    pd.DataFrame(results).to_csv(result_csv, index=False)

                    del recon_latent, recon, gt_metric, recon_metric
                    del gt_metric_for_metrics, recon_metric_for_metrics

                del gt, gt_latent, seeds_per_step, dkl_per_step
                del noisy_recons, step_indices

            model.set_branch_logging(False)
            if do_branch_log:
                for branch_row in model.consume_branch_contribution_stats(reset=True):
                    branch_row = dict(branch_row)
                    slot = int(branch_row.get("slot", -1))
                    branch_row.update({
                        "pair_index": pair_idx,
                        "pair_key": pair_key,
                        "condition_mode": condition_mode,
                        "modality": modality,
                        "gt_path": str(image_path),
                        "prompt_path": str(prompt_path),
                        "block_index": (
                            int(model.tritsp_inject_block_indices[slot])
                            if 0 <= slot < len(model.tritsp_inject_block_indices) else -1
                        ),
                    })
                    branch_results.append(branch_row)
                branch_completed_pairs.add(pair_key)
                pd.DataFrame(branch_results).to_csv(branch_csv, index=False)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                f"OOM during full {modality} evaluation of {image_path}. "
                "Run with two-GPU torchrun so RGB and thermal use separate cards, "
                "and optionally add --skip_lpips. Original error: " + str(exc)
            ) from exc
        finally:
            model.set_branch_logging(False)
            if not do_branch_log:
                model.reset_branch_contribution_stats()
            del gt_cpu
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    pd.DataFrame(results).to_csv(result_csv, index=False)
    if bool(args.branch_contribution_log):
        pd.DataFrame(branch_results).to_csv(branch_csv, index=False)


def merge_pair_results(
    output_root: Path, condition_mode: str,
    items: Sequence[PairedItemTriTSP], recon_timesteps: Sequence[int]
) -> None:
    mode_root = output_root / condition_mode
    rgb_path = mode_root / "results_rgb.csv"
    thermal_path = mode_root / "results_thermal.csv"
    if not (rgb_path.exists() and thermal_path.exists()):
        return

    rgb = pd.read_csv(rgb_path)
    thermal = pd.read_csv(thermal_path)
    keys = ["pair_key", "recon_timestep", "condition_mode"]
    merged = rgb.merge(
        thermal,
        on=keys,
        suffixes=("_rgb", "_thermal"),
        how="inner",
    )
    merged = merged.loc[merged["recon_timestep"].isin(recon_timesteps)].copy()

    packed_bits_by_key = {
        item[4]: len(pack_tsp(item[3])) * 8
        for item in items
    }
    merged["prompt_bits"] = merged["pair_key"].map(packed_bits_by_key)
    if merged["prompt_bits"].isna().any():
        missing = merged.loc[merged["prompt_bits"].isna(), "pair_key"].unique().tolist()
        raise ValueError(f"No TSP codes found for evaluated pairs: {missing}")

    rgb_pixels = merged["raw_width_rgb"] * merged["raw_height_rgb"]
    thermal_pixels = merged["raw_width_thermal"] * merged["raw_height_thermal"]
    merged["rgb_bpp"] = merged["rgb_image_bits"] / rgb_pixels
    merged["thermal_bpp"] = merged["thermal_image_bits"] / thermal_pixels
    merged["prompt_bpp"] = merged["prompt_bits"] / rgb_pixels
    merged["rgb_total_bpp"] = (
        merged["rgb_image_bits"] + merged["prompt_bits"] / 2
    ) / rgb_pixels
    merged["ir_total_bpp"] = (
        merged["thermal_image_bits"] + merged["prompt_bits"] / 2
    ) / thermal_pixels
    merged.to_csv(mode_root / "results_pair.csv", index=False)

    columns = (
        "recon_timestep", "average_prompt_bpp", "average_rgb_bpp",
        "average_thermal_bpp", "average_rgb_total_bpp", "average_ir_total_bpp"
    )
    summary_rows: List[Dict] = []
    for timestep, group in merged.groupby("recon_timestep", sort=False):
        summary_rows.append({
            "recon_timestep": int(timestep),
            "average_prompt_bpp": float(group["prompt_bpp"].mean()),
            "average_rgb_bpp": float(group["rgb_bpp"].mean()),
            "average_thermal_bpp": float(group["thermal_bpp"].mean()),
            "average_rgb_total_bpp": float(group["rgb_total_bpp"].mean()),
            "average_ir_total_bpp": float(group["ir_total_bpp"].mean()),
        })
    summary_rows.sort(key=lambda row: -row["recon_timestep"])
    pd.DataFrame(summary_rows, columns=columns).to_csv(
        mode_root / "summary_by_timestep.csv", index=False, float_format="%.8f"
    )
    print("========== Evaluate BPP Summary ==========")
    for row in summary_rows:
        print(f"\nrecon_timestep = {row['recon_timestep']}")
        for key in columns[1:]:
            print(f"{key:<25} = {row[key]:.8f}")


def merge_branch_contributions(output_root: Path, condition_mode: str) -> None:
    """Create a weighted aggregate of r_TSP/r_SPP/r_Delta across test pairs."""
    mode_root = output_root / condition_mode
    paths = [
        mode_root / "branch_contributions_rgb.csv",
        mode_root / "branch_contributions_thermal.csv",
    ]
    frames = []
    for path in paths:
        if path.exists():
            try:
                frame = pd.read_csv(path)
                if not frame.empty:
                    frames.append(frame)
            except Exception:
                pass
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True)
    required = {"condition_mode", "modality", "stage", "slot", "block_index", "count"}
    if not required.issubset(frame.columns):
        return
    metrics_cols = ["r_tsp", "r_spp", "r_delta", "gate_tsp", "gate_spp", "gate_delta"]
    rows = []
    group_cols = ["condition_mode", "modality", "stage", "slot", "block_index"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        weights = pd.to_numeric(group["count"], errors="coerce").fillna(0.0).clip(lower=0.0)
        total = float(weights.sum())
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["count"] = int(total)
        row["num_pairs"] = int(group["pair_key"].nunique()) if "pair_key" in group.columns else int(len(group))
        for col in metrics_cols:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce").fillna(0.0)
                row[col] = float((values * weights).sum() / total) if total > 0 else float("nan")
        rows.append(row)
    pd.DataFrame(rows).to_csv(mode_root / "branch_contributions_summary.csv", index=False)


def parse_args():
    p = argparse.ArgumentParser(
        description="Full TriTSP-Bridge DiffC encode/decode evaluation"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    p.add_argument("--rgb_list", required=True)
    p.add_argument("--thermal_list", "--ir_list", dest="thermal_list", required=True)
    p.add_argument(
        "--prompt_list",
        required=True,
        help="TSP JSON list; matched by last-folder/image-basename",
    )
    p.add_argument("--rgb_root", "--rgb_list_root", dest="rgb_root", default=None)
    p.add_argument(
        "--thermal_root",
        "--ir_root",
        "--ir_list_root",
        dest="thermal_root",
        default=None,
    )
    p.add_argument("--output_dir", default="./outputs/test150")
    p.add_argument("--text_cache", default=None)
    p.add_argument("--modalities", default="rgb,thermal")
    p.add_argument(
        "--recon_timesteps",
        "--recon_timestep",
        dest="recon_timesteps",
        default="610,400,170,120,70,45",
    )
    p.add_argument(
        "--preprocess_mode",
        default="pad64",
        choices=["resize", "pad64", "none"],
    )
    p.add_argument("--target_width", type=int, default=1024)
    p.add_argument("--target_height", type=int, default=576)
    p.add_argument("--compression_seed", type=int, default=0)
    p.add_argument("--skip_lpips", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--high_boundary", type=int, default=300)
    p.add_argument("--middle_boundary", type=int, default=85)
    p.add_argument("--tritsp_hidden_size", type=int, default=None)
    p.add_argument("--tritsp_cond_dim", type=int, default=512)
    p.add_argument("--tritsp_num_heads", type=int, default=8)
    p.add_argument("--tritsp_condition_tokens", type=int, default=12)
    p.add_argument("--tritsp_tsp_condition_tokens", type=int, default=8)
    p.add_argument("--tritsp_spp_condition_tokens", type=int, default=4)
    p.add_argument("--tritsp_condition_decoder_layers", type=int, default=2)
    p.add_argument("--tritsp_delta_rank", type=int, default=64)
    p.add_argument("--tritsp_gate_max", type=float, default=0.5)
    p.add_argument("--tritsp_inject_block_indices", default="9,14,18,23")
    p.add_argument("--branch_contribution_log", action="store_true", help="Measure ||g_T A_TSP||/||H||, ||g_S F_SPP||/||H||, and ||g_Delta Delta||/||H|| during evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    if args.text_cache is None:
        args.text_cache = str(Path(args.checkpoint).parent / "text_cache.pt")
    if int(args.tritsp_tsp_condition_tokens) + int(args.tritsp_spp_condition_tokens) != int(args.tritsp_condition_tokens):
        raise ValueError(
            "--tritsp_tsp_condition_tokens + --tritsp_spp_condition_tokens must equal --tritsp_condition_tokens"
        )
    saved_train_mode = checkpoint_train_mode(args.checkpoint)
    if saved_train_mode not in (None, "full"):
        raise ValueError("Only full checkpoints can be evaluated")
    modes = ["full"]
    _distributed, rank, local_rank, world_size = init_distributed()
    if world_size not in (1, 2):
        raise ValueError("TriTSP full evaluation supports one GPU or exactly two GPUs")
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    config_base = load_config(args.config)
    recon_timesteps = parse_timesteps(
        args.recon_timesteps, config_base.encoding_timesteps
    )
    config = truncate_config(config_base, recon_timesteps)

    items = build_paired_items(
        args.rgb_list, args.thermal_list, args.prompt_list
    )
    codes = [item[3] for item in items]
    rgb_paths = [item[0] for item in items]
    thermal_paths = [item[1] for item in items]
    if rank == 0:
        write_pair_manifest(
            items, output_root / "evaluation_pair_manifest.csv"
        )

    model = make_model(config, args, device)
    cache_path = Path(args.text_cache)
    if rank == 0:
        if cache_path.exists():
            try:
                model.tritsp_text_cache = TriTSPTextCache.load(cache_path)
            except ValueError:
                model.tritsp_text_cache = TriTSPTextCache(num_stage_tokens=8, num_field_tokens=2)
        model.prebuild_text_cache(codes, cache_path)
    barrier()
    model.tritsp_text_cache = TriTSPTextCache.load(cache_path)

    model.prepare_base_prompt_cache([
        config.encoding_guidance_scale,
        config.denoising_guidance_scale,
    ])
    model.offload_text_encoder()
    model.tritsp_adapter.eval()

    simulator = GaussianChannelSimulator(
        config.max_chunk_size, config.chunk_padding
    )
    requested = [
        x.strip().lower() for x in args.modalities.split(",") if x.strip()
    ]
    requested = [
        "thermal" if x in {"thermal", "ir", "t"} else "rgb"
        for x in requested
    ]
    if world_size == 2:
        assigned = ["rgb"] if rank == 0 else ["thermal"]
        assigned = [m for m in assigned if m in requested]
    else:
        assigned = [m for m in ("rgb", "thermal") if m in requested]

    rgb_root = get_root(rgb_paths, args.rgb_root)
    thermal_root = get_root(thermal_paths, args.thermal_root)

    if rank == 0:
        with open(
            output_root / "evaluation_config.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "args": vars(args),
                    "recon_timesteps": recon_timesteps,
                    "num_pairs": len(items),
                    "condition_modes": modes,
                    "checkpoint_train_condition_mode": saved_train_mode,
                    "thermal_preprocess": (
                        "single-channel copied directly to RGB; same pad64 as RGB; "
                        "no percentile normalization"
                    ),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    for condition_mode in modes:
        for modality in assigned:
            evaluate_modality(
                model=model,
                modality=modality,
                items=items,
                image_root=rgb_root if modality == "rgb" else thermal_root,
                output_root=output_root,
                condition_mode=condition_mode,
                config=config,
                args=args,
                simulator=simulator,
                rank=rank,
            )
        barrier()
        if rank == 0:
            merge_pair_results(output_root, condition_mode, items, recon_timesteps)
            if bool(args.branch_contribution_log):
                merge_branch_contributions(output_root, condition_mode)
        barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
