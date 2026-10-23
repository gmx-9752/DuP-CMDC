# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Sequence, Tuple

from lib.dup_cmdc.tsp_core import (
    load_tsp_core_json,
    read_path_list,
)

IMAGE_SUFFIXES_DUP_CMDC = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


MODALITY_DIR_NAMES_DUP_CMDC = {
    "rgb", "visible", "vis", "ir", "thermal", "infrared", "tir", "lwir",
}
PairedItemDuPCmdc = Tuple[Path, Path, Path, Dict[str, Any], str]


class PairingReportDuPCmdc(NamedTuple):
    rgb_raw_count: int
    thermal_raw_count: int
    prompt_raw_count: int
    matched_count: int
    missing_from_rgb: Tuple[str, ...]
    missing_from_thermal: Tuple[str, ...]
    missing_from_prompt: Tuple[str, ...]


def _basename_without_prompt_suffix(path: Path, kind: str) -> str:
    name = path.name
    if str(kind).lower() == "prompt":
        if name.lower().endswith(".json"):
            name = name[:-5]
        lower = name.lower()
        for suffix in sorted(IMAGE_SUFFIXES_DUP_CMDC, key=len, reverse=True):
            if lower.endswith(suffix):
                name = name[: -len(suffix)]
                break
    else:
        name = path.stem
    return Path(name).stem


def pair_key_from_path(path: str | Path, kind: str) -> str:
    p = Path(path)
    stem = _basename_without_prompt_suffix(p, kind)
    parent = p.parent
    if parent.name.lower() in MODALITY_DIR_NAMES_DUP_CMDC and parent.parent.name:
        scene = parent.parent.name
    else:
        scene = parent.name
    if not scene or not stem:
        raise ValueError(f"Cannot build scene/filename key from path: {p}")
    return f"{scene}/{stem}"


def _build_unique_map(paths: Sequence[Path], kind: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    duplicates: Dict[str, List[str]] = {}
    for path in paths:
        key = pair_key_from_path(path, kind)
        if key in mapping:
            duplicates.setdefault(key, [str(mapping[key])]).append(str(path))
        else:
            mapping[key] = path
    if duplicates:
        preview = "; ".join(f"{key}: {values[:3]}" for key, values in list(duplicates.items())[:10])
        raise ValueError(f"Duplicate {kind} canonical keys detected: {preview}")
    return mapping


def inspect_pair_lists(
    rgb_list: str | Path,
    thermal_list: str | Path,
    prompt_list: str | Path,
    *,
    require_files: bool = True,
) -> Tuple[Dict[str, Path], Dict[str, Path], Dict[str, Path], PairingReportDuPCmdc]:
    rgb_paths = read_path_list(rgb_list)
    thermal_paths = read_path_list(thermal_list)
    prompt_paths = read_path_list(prompt_list)
    if require_files:
        missing_files = [str(p) for p in rgb_paths + thermal_paths + prompt_paths if not p.is_file()]
        if missing_files:
            raise FileNotFoundError("List contains missing files: " + ", ".join(missing_files[:20]))
    rgb_map = _build_unique_map(rgb_paths, "rgb")
    thermal_map = _build_unique_map(thermal_paths, "thermal")
    prompt_map = _build_unique_map(prompt_paths, "prompt")
    rgb_keys, thermal_keys, prompt_keys = set(rgb_map), set(thermal_map), set(prompt_map)
    common = rgb_keys & thermal_keys & prompt_keys
    report = PairingReportDuPCmdc(
        len(rgb_paths), len(thermal_paths), len(prompt_paths), len(common),
        tuple(sorted((thermal_keys | prompt_keys) - rgb_keys)),
        tuple(sorted((rgb_keys | prompt_keys) - thermal_keys)),
        tuple(sorted((rgb_keys | thermal_keys) - prompt_keys)),
    )
    return rgb_map, thermal_map, prompt_map, report


def write_unmatched_report(report: PairingReportDuPCmdc, path: str | Path, split_name: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(f"split={split_name}\n")
        handle.write(f"rgb_raw_count={report.rgb_raw_count}\n")
        handle.write(f"thermal_raw_count={report.thermal_raw_count}\n")
        handle.write(f"prompt_raw_count={report.prompt_raw_count}\n")
        handle.write(f"matched_count={report.matched_count}\n\n")
        for label, values in (
            ("missing_from_rgb", report.missing_from_rgb),
            ("missing_from_thermal", report.missing_from_thermal),
            ("missing_from_prompt", report.missing_from_prompt),
        ):
            handle.write(f"[{label}]\n")
            handle.writelines(f"{value}\n" for value in values)
            handle.write("\n")


def build_paired_items(
    rgb_list: str | Path,
    thermal_list: str | Path,
    prompt_list: str | Path,
    *,
    require_files: bool = True,
    unmatched_path: str | Path | None = None,
    split_name: str = "",
) -> List[PairedItemDuPCmdc]:
    rgb_map, thermal_map, prompt_map, report = inspect_pair_lists(
        rgb_list, thermal_list, prompt_list, require_files=require_files
    )
    if unmatched_path is not None:
        write_unmatched_report(report, unmatched_path, split_name)
    common = set(rgb_map) & set(thermal_map) & set(prompt_map)
    items: List[PairedItemDuPCmdc] = []
    for key in sorted(common):
        prompt_path = prompt_map[key]
        codes = load_tsp_core_json(prompt_path)
        items.append((rgb_map[key], thermal_map[key], prompt_path, codes, key))
    if not items:
        raise ValueError("No complete RGB/Thermal/TSP canonical pairs were matched")
    return items


def write_pair_manifest(items: Sequence[PairedItemDuPCmdc], output_csv: str | Path) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_index", "pair_key", "rgb_path", "thermal_path", "prompt_path",
        "geometry", "environment", "human_present", "human_organization", "human_regions",
        "human_topology", "human_morphology", "vehicle_present", "vehicle_organization",
        "vehicle_regions", "vehicle_topology", "vehicle_morphology",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (rgb, thermal, prompt, codes, key) in enumerate(items):
            content = codes["content"]
            row = {
                "pair_index": index, "pair_key": key, "rgb_path": str(rgb),
                "thermal_path": str(thermal), "prompt_path": str(prompt),
                "geometry": codes["geometry"], "environment": ";".join(codes["environment"]),
            }
            for obj_key in ("human", "vehicle"):
                branch = content[obj_key]
                row.update({
                    f"{obj_key}_present": int(branch["present"]),
                    f"{obj_key}_organization": branch.get("organization", ""),
                    f"{obj_key}_regions": ";".join(branch.get("regions", [])),
                    f"{obj_key}_topology": branch.get("topology", ""),
                    f"{obj_key}_morphology": ";".join(branch.get("morphology", [])),
                })
            writer.writerow(row)


PairedItemTriTSP = PairedItemDuPCmdc
