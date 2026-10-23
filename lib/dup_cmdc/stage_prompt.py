from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List, Sequence, Tuple

STAGE_PROMPTS_DUP_CMDC: Dict[str, Dict[str, str]] = {
    "high": {
        "shared": (
            "Preserve global scene topology, foreground existence, coarse object layout, "
            "and coarse foreground-background separation. Avoid global structure collapse, "
            "missing foreground objects, merged major regions, and uniform structural blur."
        ),
        "rgb": (
            "Preserve visible object silhouettes, coarse structural contours, and major visible "
            "boundaries. Avoid broken visible geometry, washed-out contours, and large-scale color collapse."
        ),
        "thermal": (
            "Preserve compact thermal target existence, coarse heat-source separation, and coarse "
            "thermal contrast. Avoid missing weak thermal targets, merged hotspots, diffuse blobs, "
            "thermal bleeding, and uniform thermal patches."
        ),
    },
    "middle": {
        "shared": (
            "Refine object boundaries and preserve structurally faithful contours, small foreground "
            "target shapes, and foreground-background separation. Avoid boundary smearing and adjacent target merging."
        ),
        "rgb": (
            "Recover visible edge sharpness, local structural transitions, thin visible structures, "
            "and faithful color boundaries. Avoid edge washout, broken thin structures, color spillage, "
            "unnatural color blocks, and desaturation."
        ),
        "thermal": (
            "Sharpen thermal object boundaries, preserve compact hotspot shapes, recover weak meaningful "
            "thermal foreground responses, and restore thermal contrast. Avoid halo expansion, boundary blur, "
            "thermal clutter confusion, diffuse patches, and noise amplification."
        ),
    },
    "low": {
        "shared": (
            "Recover fine structural details and local boundary sharpness while preserving subtle foreground "
            "details and fine linear structures. Avoid oversmoothing, ringing, artificial high-frequency noise, "
            "local texture loss, and smearing of fine linear features."
        ),
        "rgb": (
            "Recover natural visible texture transitions, thin visible details, faithful color gradients, and "
            "detailed surface patterns. Avoid color banding, posterization, texture washout, edge ringing, flat "
            "color regions, block artifacts, and color spillage."
        ),
        "thermal": (
            "Preserve fine thermal boundaries, compact thermal contrast, subtle thermal details, and distinct "
            "compact hotspots without amplifying noise. Avoid thermal speckle, bleed, over-expanded hotspots, "
            "blurred fine boundaries, and faint or missing compact hotspots."
        ),
    },
}

STAGE_TO_INDEX_DUP_CMDC = {"high": 0, "middle": 1, "low": 2}


def stage_from_timestep(timestep: int, high_boundary: int = 300, middle_boundary: int = 85) -> str:
    t = int(timestep)
    if t >= int(high_boundary):
        return "high"
    if t >= int(middle_boundary):
        return "middle"
    return "low"


def stage_prompt_text(stage: str, modality: str) -> str:
    key = str(stage).lower()
    if key not in STAGE_PROMPTS_DUP_CMDC:
        raise ValueError(f"Unknown stage: {stage!r}")
    mod = str(modality).lower()
    if mod in {"ir", "thermal", "t"}:
        mod = "thermal"
    elif mod in {"rgb", "visible", "r"}:
        mod = "rgb"
    else:
        raise ValueError(f"Unknown modality: {modality!r}")
    p = STAGE_PROMPTS_DUP_CMDC[key]
    return p["shared"] + "\n\n" + p[mod]


def build_path_edges_to_stop(encoding_timesteps: Sequence[int], stop_timestep: int = 45) -> List[Tuple[int, int, int]]:
    schedule = [int(t) for t in encoding_timesteps]
    stop = int(stop_timestep)
    if stop not in schedule:
        nearest = sorted(schedule, key=lambda x: abs(x - stop))[:10]
        raise ValueError(f"stop_timestep={stop} is not in encoding_timesteps; nearest={nearest}")
    schedule = schedule[: schedule.index(stop) + 1]
    current = [1000] + schedule[:-1]
    return [(int(a), int(b), int(i)) for i, (a, b) in enumerate(zip(current, schedule))]


def split_edges_by_stage(
    edges: Iterable[Tuple[int, int, int]],
    high_boundary: int = 300,
    middle_boundary: int = 85,
) -> Dict[str, List[Tuple[int, int, int]]]:
    out = {"high": [], "middle": [], "low": []}
    for edge in edges:
        out[stage_from_timestep(edge[0], high_boundary, middle_boundary)].append(edge)
    for stage, values in out.items():
        if not values:
            raise ValueError(f"No fixed-path edges assigned to stage={stage}")
    return out


def representative_edges(stage_edges: Dict[str, Sequence[Tuple[int, int, int]]], per_stage: int = 1):
    selected = []
    n_pick = max(1, int(per_stage))
    for stage in ("high", "middle", "low"):
        values = list(stage_edges[stage])
        if n_pick == 1:
            idxs = [len(values) // 2]
        else:
            idxs = [round((k + 1) * (len(values) - 1) / (n_pick + 1)) for k in range(n_pick)]
        selected.extend((stage, values[int(i)]) for i in idxs)
    return selected


def stage_prompt_sha256() -> str:
    prompts = [
        stage_prompt_text(stage, modality)
        for stage in ("high", "middle", "low")
        for modality in ("rgb", "thermal")
    ]
    payload = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
