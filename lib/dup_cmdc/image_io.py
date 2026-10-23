# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image

from lib import image_utils
from lib.uav_preprocess import preprocess_uav_pil


def _preprocess_rgb_pil(img: Image.Image, mode: str, target_width: int, target_height: int):
    return preprocess_uav_pil(
        img.convert("RGB"),
        mode=mode,
        target_width=int(target_width),
        target_height=int(target_height),
    )


def load_rgb_tensor(
    path: str | Path,
    preprocess_mode: str = "pad64",
    target_width: int = 1024,
    target_height: int = 576,
):
    p = Path(path)
    with Image.open(p) as opened:
        raw = opened.convert("RGB")
    raw_w, raw_h = raw.size
    proc, meta = _preprocess_rgb_pil(raw, preprocess_mode, target_width, target_height)
    tensor = image_utils.pil_to_torch_img(proc).float()
    meta = dict(meta)
    meta.update({"source_mode": "RGB", "raw_path": str(p)})
    return tensor, (raw_h, raw_w), meta


def load_thermal_tensor(
    path: str | Path,
    preprocess_mode: str = "pad64",
    target_width: int = 1024,
    target_height: int = 576,
):
    """Copy a single-channel thermal frame to RGB and apply RGB ``pad64``.

    No percentile, min-max, histogram, or per-image contrast normalization is
    performed.  For the VT-UAV 8-bit thermal images, ``convert('L')`` preserves
    the 0--255 grayscale convention before the three channels are duplicated.
    """
    p = Path(path)
    with Image.open(p) as opened:
        source_mode = opened.mode
        thermal_l = opened.convert("L")
    raw_w, raw_h = thermal_l.size
    thermal_rgb = Image.merge("RGB", (thermal_l, thermal_l, thermal_l))
    proc, meta = _preprocess_rgb_pil(
        thermal_rgb, preprocess_mode, target_width, target_height
    )
    tensor3 = image_utils.pil_to_torch_img(proc).float()
    
    
    gray1 = tensor3[:, :1] if tensor3.ndim == 4 else tensor3[:1]
    meta = dict(meta)
    meta.update({
        "source_mode": source_mode,
        "thermal_conversion": "PIL_L_direct_repeat3_no_percentile",
        "raw_path": str(p),
    })
    return tensor3, gray1, (raw_h, raw_w), meta


def crop_to_raw(tensor: torch.Tensor, raw_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = int(raw_hw[0]), int(raw_hw[1])
    return tensor[..., :h, :w]


def gray3(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[-3] == 1:
        return tensor.repeat_interleave(3, dim=-3)
    gray = tensor.mean(dim=-3, keepdim=True)
    return gray.repeat_interleave(3, dim=-3)
