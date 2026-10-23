# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import random
from typing import Any, Dict, Mapping, Tuple

from lib.dup_cmdc.tsp_core import normalize_tsp_core
from lib.dup_cmdc.tsp_schema import (
    GEOMETRY_CODES, ORGANIZATION_CODES, REGION_CODES, TOPOLOGY_CODES, MORPHOLOGY_CODES,
)


def _change_one(value: str, values) -> str:
    choices = [str(x) for x in values if str(x) != str(value)]
    return choices[0] if choices else str(value)


def _signature(c: Mapping[str, Any]) -> Dict[str, Any]:
    value = normalize_tsp_core(c)
    out: Dict[str, Any] = {"geometry": value["geometry"], "environment": set(value["environment"])}
    for key in ("human", "vehicle"):
        branch = value["content"][key]
        out[key] = (
            bool(branch["present"]), branch.get("organization"), tuple(branch.get("regions", [])),
            branch.get("topology"), tuple(branch.get("morphology", [])),
        )
    return out


def semantic_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    aa, bb = _signature(a), _signature(b)
    score = 0.0
    score += float(aa["geometry"] != bb["geometry"])
    env_a, env_b = aa["environment"], bb["environment"]
    score += 0.5 * (1.0 - len(env_a & env_b) / max(1, len(env_a | env_b)))
    for key in ("human", "vehicle"):
        pa, pb = aa[key], bb[key]
        score += float(pa[0] != pb[0])
        if pa[0] and pb[0]:
            score += float(pa[1] != pb[1])
            score += float(set(pa[2]) != set(pb[2]))
            score += float(pa[3] != pb[3])
            score += float(set(pa[4]) != set(pb[4]))
    return score / 9.0


def _candidate_pool(correct: Mapping[str, Any]) -> list[Dict[str, Any]]:
    base = normalize_tsp_core(correct)
    candidates: list[Dict[str, Any]] = []
    for code in GEOMETRY_CODES:
        if code != base["geometry"]:
            item = copy.deepcopy(base); item["geometry"] = code; candidates.append(item)
    for key in ("human", "vehicle"):
        branch = base["content"][key]
        if not branch["present"]:
            continue
        for code in ORGANIZATION_CODES:
            if code != branch["organization"]:
                item = copy.deepcopy(base); item["content"][key]["organization"] = code; candidates.append(item)
        for code in TOPOLOGY_CODES:
            if code != branch["topology"]:
                item = copy.deepcopy(base); item["content"][key]["topology"] = code; candidates.append(item)
        for code in REGION_CODES:
            if code not in branch["regions"]:
                item = copy.deepcopy(base); item["content"][key]["regions"] = [code]; candidates.append(item)
        for code in MORPHOLOGY_CODES:
            if code not in branch["morphology"]:
                item = copy.deepcopy(base); item["content"][key]["morphology"] = [code]; candidates.append(item)
    return [normalize_tsp_core(item) for item in candidates]


def generate_counterfactual(
    correct: Mapping[str, Any], mode: str, seed: int,
    far_min_distance: float = 0.35, hard_min_distance: float = 0.08, hard_max_distance: float = 0.35,
) -> Dict[str, Any]:
    corr = normalize_tsp_core(correct)
    candidates = _candidate_pool(corr)
    if not candidates:
        return corr
    scored = [(semantic_distance(corr, candidate), candidate) for candidate in candidates]
    mode_key = str(mode).lower()
    if mode_key == "hard":
        selected = [(d, c) for d, c in scored if hard_min_distance <= d <= hard_max_distance]
    elif mode_key == "far":
        selected = [(d, c) for d, c in scored if d >= far_min_distance]
    else:
        selected = scored
    if not selected:
        selected = scored
    rng = random.Random(int(seed))
    rng.shuffle(selected)
    
    
    if mode_key == "hard":
        best = min(d for d, _ in selected)
    else:
        best = max(d for d, _ in selected)
    ties = [c for d, c in selected if abs(d - best) < 1e-9]
    return copy.deepcopy(ties[0] if ties else selected[0][1])


def negative_mode_for_epoch(epoch: int, seed: int) -> str:
    e = int(epoch)
    if e < 5:
        return "none"
    if e < 15:
        return "far"
    return "hard" if random.Random(int(seed)).random() < 0.70 else "far"


def changed_fields(a: Mapping[str, Any], b: Mapping[str, Any]) -> int:
    aa, bb = _signature(a), _signature(b)
    fields = ["geometry", "environment", "human", "vehicle"]
    return sum(aa[f] != bb[f] for f in fields)

