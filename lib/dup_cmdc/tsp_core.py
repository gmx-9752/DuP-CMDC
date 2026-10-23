# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from lib.dup_cmdc.tsp_schema import (
    CONTENT_CODES,
    ENVIRONMENT_CODES,
    GEOMETRY_CODES,
    MORPHOLOGY_CODES,
    ORGANIZATION_CODES,
    REGION_CODES,
    TOPOLOGY_CODES,
    canonicalize_codes,
    codebook_bit_cost,
    render_tsp_prompts,
)

TSP_VERSION_DUP_CMDC = "TSP-Codebook v7 / G-E-C + O-R-T-M / Semantic-R10"
OBJECT_KEYS_DUP_CMDC: Tuple[str, ...] = ("human", "vehicle")
FIELD_TYPE_IDS_DUP_CMDC = {"G": 0, "E": 1, "C": 2, "O": 3, "R": 4, "T": 5, "M": 6}


def _unwrap_mapping(obj: Mapping[str, Any]) -> Mapping[str, Any]:
    root: Mapping[str, Any] = obj
    for key in ("thermal_side_prompt", "tsp", "codes", "prediction", "result"):
        value = root.get(key) if isinstance(root, Mapping) else None
        if isinstance(value, Mapping):
            root = value
    return root


def normalize_tsp_core(codes: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(codes, Mapping):
        raise TypeError(f"TSP codes must be a mapping, got {type(codes)!r}")
    return canonicalize_codes(_unwrap_mapping(codes), enable_train_slot=False)




def load_tsp_core_json(path: str | Path) -> Dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"TSP JSON must contain an object: {path}")
    return normalize_tsp_core(value)


def read_path_list(path: str | Path) -> List[Path]:
    list_path = Path(path).expanduser().resolve()
    values: List[Path] = []
    with open(list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            candidate = Path(text.split()[0]).expanduser()
            values.append((list_path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
    return values


def tsp_side_information_bits(codes: Mapping[str, Any]) -> int:
    """Shared pair-side-information cost: 10 + 17 per active object."""
    return int(codebook_bit_cost(normalize_tsp_core(codes)))


def core_bits(codes: Mapping[str, Any] | None = None) -> int:
    return 10 if codes is None else tsp_side_information_bits(codes)


def _code_description(field_type: str, code: str, modality: str) -> str:
    mod = "ir" if str(modality).lower() in {"ir", "thermal", "t"} else "rgb"
    if field_type == "G":
        return GEOMETRY_CODES[code]["render"]
    if field_type == "E":
        return ENVIRONMENT_CODES[code][mod]
    if field_type == "C":
        family_code = code[2:] if code.startswith("C_") else code
        return CONTENT_CODES[family_code]["en"]
    if field_type == "O":
        return ORGANIZATION_CODES[code]["render"]
    if field_type == "R":
        return REGION_CODES[code]["render"]
    if field_type == "T":
        return TOPOLOGY_CODES[code]["render"]
    if field_type == "M":
        return MORPHOLOGY_CODES[code][mod]
    raise KeyError(f"Unknown TSP primitive type: {field_type}")


def iter_tsp_primitive_specs(codes: Mapping[str, Any]) -> List[Tuple[str, str, str, str]]:
    c = normalize_tsp_core(codes)
    out: List[Tuple[str, str, str, str]] = [(c["geometry"], c["geometry"], "G", "")]
    out.extend((code, code, "E", "") for code in c["environment"])
    for obj_key in OBJECT_KEYS_DUP_CMDC:
        branch = c["content"][obj_key]
        if branch["present"]:
            prefix = "H" if obj_key == "human" else "V"
            out.append((f"C_{prefix}", f"C_{prefix}", "C", obj_key))
    for obj_key in OBJECT_KEYS_DUP_CMDC:
        branch = c["content"][obj_key]
        if not branch["present"]:
            continue
        prefix = "H" if obj_key == "human" else "V"
        out.append((f"{prefix}_{branch['organization']}", branch["organization"], "O", obj_key))
        out.extend((f"{prefix}_{code}", code, "R", obj_key) for code in branch["regions"])
        out.append((f"{prefix}_{branch['topology']}", branch["topology"], "T", obj_key))
        out.extend((f"{prefix}_{code}", code, "M", obj_key) for code in branch["morphology"])
    return out




def primitive_text(token_key: str, code: str, field_type: str, object_key: str, modality: str) -> str:
    mod = "thermal" if str(modality).lower() in {"thermal", "ir", "t"} else "rgb"
    if field_type == "C":
        family = "human" if token_key.endswith("H") else "vehicle"
        return f"[{token_key}] content presence: {family} salient objects are present."
    desc = _code_description(field_type, code, mod)
    owner = f" {object_key}" if object_key else ""
    return f"[{token_key}]{owner} {field_type}-field={code}: {desc}."


def render_tsp_core(codes: Mapping[str, Any], modality: str) -> str:
    rendered = render_tsp_prompts(normalize_tsp_core(codes))
    return rendered["ir" if str(modality).lower() in {"ir", "thermal", "t"} else "rgb"]




__all__ = [
    "TSP_VERSION_DUP_CMDC", "OBJECT_KEYS_DUP_CMDC", "FIELD_TYPE_IDS_DUP_CMDC",
    "normalize_tsp_core",
    "load_tsp_core_json", "read_path_list",
    "tsp_side_information_bits", "core_bits",
    "iter_tsp_primitive_specs",
    "primitive_text", "render_tsp_core",
]
