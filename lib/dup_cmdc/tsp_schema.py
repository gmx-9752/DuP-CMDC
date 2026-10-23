#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

VERSION = "TSP-Codebook v7 / Structured Sparse Thermal Semantic Prompt / Region-Only Semantic-R10 CoverageFirst-v7"


GEOMETRY_CODES: Dict[str, Dict[str, str]] = {
    "G0": {"en": "open_or_weakly_structured_layout", "zh": "开放、平滑或弱结构布局", "render": "open or weakly structured"},
    "G1": {"en": "linear_or_corridor_layout", "zh": "直线道路、桥梁或走廊式布局", "render": "linear or corridor-like"},
    "G2": {"en": "crossing_branching_or_network_layout", "zh": "交叉、分支或网络布局", "render": "crossing, branching, or network-like"},
    "G3": {"en": "regular_enclosed_or_block_layout", "zh": "规则块状、围合或街区式布局", "render": "regular enclosed or block-like"},
    "G4": {"en": "curved_path_or_boundary_layout", "zh": "弯道、曲线路径或曲线边界布局", "render": "curved-path or curved-boundary"},
    "G5": {"en": "irregular_or_cluttered_layout", "zh": "杂乱或不规则布局", "render": "irregular or cluttered"},
}

ENVIRONMENT_CODES: Dict[str, Dict[str, str]] = {
    "E_B": {"en": "built_structures", "zh": "建筑、屋顶、墙体或规则人工结构", "rgb": "built structures", "ir": "large-scale built boundaries"},
    "E_P": {"en": "paved_or_hard_surface", "zh": "道路、停车区、广场等铺装或硬质地面", "rgb": "paved or hard-surface regions", "ir": "paved or hard-surface thermal regions"},
    "E_V": {"en": "vegetation", "zh": "树木、草地或绿化", "rgb": "vegetation regions", "ir": "vegetation-related thermal background"},
    "E_W": {"en": "water", "zh": "海面、河流或水体", "rgb": "water regions", "ir": "water-related thermal regions"},
    "E_G": {"en": "bare_or_soft_ground", "zh": "裸地、沙地或土地区域", "rgb": "bare or soft-ground regions", "ir": "bare or soft-ground thermal regions"},
}

CONTENT_CODES: Dict[str, Dict[str, str]] = {
    "H": {"en": "human_like_salient_objects", "zh": "人/行人类显著目标"},
    "V": {"en": "vehicle_like_salient_objects", "zh": "普通道路车辆类显著目标"},
    "T": {"en": "large_transport_or_train_like_salient_objects", "zh": "大型运输/列车类显著目标"},
}

OBJECT_FAMILY_KEYS = {"H": "human", "V": "vehicle", "T": "train"}
OBJECT_KEY_TO_CODE = {v: k for k, v in OBJECT_FAMILY_KEYS.items()}
CORE_OBJECT_KEYS = ("human", "vehicle")
OPTIONAL_OBJECT_KEYS = ("train",)

ORGANIZATION_CODES: Dict[str, Dict[str, str]] = {
    "O1": {"en": "single_or_few_isolated_instances", "zh": "单个或少量孤立目标", "render": "single or a few isolated"},
    "O2": {"en": "multiple_sparse_instances", "zh": "多个稀疏目标", "render": "multiple sparse"},
    "O3": {"en": "dense_or_crowded_instances", "zh": "密集或大量目标", "render": "dense or crowded"},
    "O4": {"en": "sparse_and_dense_coexist", "zh": "稀疏与局部密集目标同时存在", "render": "mixed sparse and locally dense"},
}


REGION_CODES: Dict[str, Dict[str, str]] = {
    
    
    "R1": {"en": "upper_horizontal_band", "zh": "上部横向区域", "render": "upper"},
    "R2": {"en": "middle_vertical_band", "zh": "中部纵向区域", "render": "middle"},
    "R3": {"en": "lower_horizontal_band", "zh": "下部横向区域", "render": "lower"},
    "R4": {"en": "right_vertical_band", "zh": "右侧纵向区域", "render": "right-side"},
    "R5": {"en": "left_vertical_band", "zh": "左侧纵向区域", "render": "left-side"},
    "R6": {"en": "compact_central_core", "zh": "中心集中区域", "render": "central"},
    
    
    "R7": {"en": "compact_upper_left_corner", "zh": "明显左上角集中", "render": "upper-left"},
    "R8": {"en": "compact_lower_left_corner", "zh": "明显左下角集中", "render": "lower-left"},
    "R9": {"en": "compact_upper_right_corner", "zh": "明显右上角集中", "render": "upper-right"},
    "R10": {"en": "compact_lower_right_corner", "zh": "明显右下角集中", "render": "lower-right"},
}

TOPOLOGY_CODES: Dict[str, Dict[str, str]] = {
    "T1": {"en": "localized_or_concentrated", "zh": "集中在单一区域或局部范围", "render": "localized or concentrated"},
    "T2": {"en": "linear_or_corridor_distribution", "zh": "沿道路、通道或主方向线性排列", "render": "distributed along a linear or corridor-like direction"},
    "T3": {"en": "multiple_separated_regions", "zh": "分布于多个明显分离区域", "render": "distributed across multiple separated regions"},
    "T4": {"en": "widespread_distribution", "zh": "大范围分散于整幅图", "render": "widely distributed across the scene"},
}

MORPHOLOGY_CODES: Dict[str, Dict[str, str]] = {
    
    
    "M0": {"en": "no_reliable_morphology_declared", "zh": "形态无法可靠判断", "rgb": "no reliable morphology declared", "ir": "no reliable thermal morphology declared"},
    "M1": {"en": "point_like_response", "zh": "点状或极小斑点热响应", "rgb": "tiny point-like appearances", "ir": "point-like thermal responses"},
    "M2": {"en": "compact_blob_like_response", "zh": "紧凑块状热响应", "rgb": "compact blob-like silhouettes", "ir": "compact blob-like thermal responses"},
    "M3": {"en": "elongated_or_slender_response", "zh": "细长、条带或延展热响应", "rgb": "elongated or slender appearances", "ir": "elongated or slender thermal responses"},
}


FIELD_ORDER = ["geometry", "environment", "content", "human", "vehicle", "train"]


GEOMETRY_BITS = 3
ENVIRONMENT_BITS = 5
CONTENT_CORE_BITS = 2  
ORGANIZATION_BITS = 2
REGION_BITS = 10
TOPOLOGY_BITS = 2
MORPHOLOGY_BITS = 3
OBJECT_BRANCH_BITS = ORGANIZATION_BITS + REGION_BITS + TOPOLOGY_BITS + MORPHOLOGY_BITS  

_GEOMETRY_INDEX = {c: i for i, c in enumerate(GEOMETRY_CODES)}
_INDEX_GEOMETRY = {i: c for c, i in _GEOMETRY_INDEX.items()}
_ORG_INDEX = {c: i for i, c in enumerate(ORGANIZATION_CODES)}
_INDEX_ORG = {i: c for c, i in _ORG_INDEX.items()}
_TOPOLOGY_INDEX = {c: i for i, c in enumerate(TOPOLOGY_CODES)}
_INDEX_TOPOLOGY = {i: c for c, i in _TOPOLOGY_INDEX.items()}
_ENV_ORDER = list(ENVIRONMENT_CODES.keys())
_REGION_ORDER = list(REGION_CODES.keys())
_MORPH_ORDER = list(MORPHOLOGY_CODES.keys())


def active_object_keys(enable_train_slot: bool = False) -> Tuple[str, ...]:
    return CORE_OBJECT_KEYS + (OPTIONAL_OBJECT_KEYS if enable_train_slot else ())


def _upper_code(value: Any) -> str:
    return str(value).strip().upper()


def _unique_codes(values: Iterable[Any], allowed: Mapping[str, Any], *, field: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        code = _upper_code(value)
        if code not in allowed:
            raise ValueError(f"invalid {field} code: {value!r}")
        if code not in seen:
            out.append(code)
            seen.add(code)
    return out


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "present"}:
            return True
        if s in {"false", "0", "no", "absent"}:
            return False
    raise ValueError(f"cannot parse boolean from {value!r}")


def empty_object_branch() -> Dict[str, Any]:
    return {"present": False}


def canonicalize_codes(obj: Mapping[str, Any], *, enable_train_slot: bool = False) -> Dict[str, Any]:
    root: Mapping[str, Any] = obj
    if isinstance(obj.get("thermal_side_prompt"), Mapping):
        root = obj["thermal_side_prompt"]  
    if isinstance(root.get("codes"), Mapping):
        root = root["codes"]  

    geometry_raw = root.get("geometry", root.get("G", root.get("g")))
    if geometry_raw is None:
        raise ValueError("missing geometry")
    geometry = _upper_code(geometry_raw)
    if geometry not in GEOMETRY_CODES:
        raise ValueError(f"invalid geometry: {geometry_raw!r}")

    env_raw = root.get("environment", root.get("E", root.get("e", [])))
    if env_raw is None:
        env_raw = []
    if isinstance(env_raw, str):
        env_raw = [x for x in env_raw.replace(";", ",").split(",") if x.strip()]
    if not isinstance(env_raw, Sequence):
        raise ValueError("environment must be a list of E_* codes")
    environment = _unique_codes(env_raw, ENVIRONMENT_CODES, field="environment")
    environment.sort(key=_ENV_ORDER.index)

    content_raw = root.get("content", root.get("C", root.get("c", {})))
    if content_raw is None:
        content_raw = {}


    
    compact_present: set[str] = set()
    if isinstance(content_raw, Sequence) and not isinstance(content_raw, (str, bytes)):
        for item in content_raw:
            code = _upper_code(item)
            if code in OBJECT_FAMILY_KEYS:
                compact_present.add(OBJECT_FAMILY_KEYS[code])
            elif str(item).strip().lower() in OBJECT_KEY_TO_CODE:
                compact_present.add(str(item).strip().lower())
            else:
                raise ValueError(f"invalid content family: {item!r}")
        content_map: Mapping[str, Any] = {}
    elif isinstance(content_raw, Mapping):
        content_map = content_raw
    else:
        raise ValueError("content must be an object keyed by human/vehicle[/train]")

    content: Dict[str, Any] = {}
    for obj_key in active_object_keys(enable_train_slot):
        branch_raw = content_map.get(obj_key, root.get(obj_key))
        if branch_raw is None:
            
            if obj_key in compact_present:
                raise ValueError(f"content marks {obj_key} present but branch details are missing")
            content[obj_key] = empty_object_branch()
            continue
        if isinstance(branch_raw, bool):
            present = bool(branch_raw)
            branch_map: Mapping[str, Any] = {"present": present}
        elif isinstance(branch_raw, Mapping):
            branch_map = branch_raw
            present_raw = branch_map.get("present", obj_key in compact_present)
            present = _parse_bool(present_raw)
        else:
            raise ValueError(f"content.{obj_key} must be an object")

        if not present:
            forbidden_keys = {"organization", "regions", "region", "topology", "morphology", "O", "R", "T", "M", "o", "r", "t", "m"}
            supplied = [k for k in forbidden_keys if k in branch_map and branch_map.get(k) not in (None, [], "")]
            if supplied:
                raise ValueError(
                    f"content.{obj_key} has present=false but still contains object attributes: {sorted(supplied)}"
                )
            content[obj_key] = empty_object_branch()
            continue

        organization_raw = branch_map.get("organization", branch_map.get("O", branch_map.get("o")))
        topology_raw = branch_map.get("topology", branch_map.get("T", branch_map.get("t")))
        regions_raw = branch_map.get("regions", branch_map.get("region", branch_map.get("R", branch_map.get("r"))))
        morph_raw = branch_map.get("morphology", branch_map.get("M", branch_map.get("m", [])))
        if organization_raw is None:
            raise ValueError(f"content.{obj_key}.organization is required when present=true")
        if topology_raw is None:
            raise ValueError(f"content.{obj_key}.topology is required when present=true")
        if regions_raw is None:
            raise ValueError(f"content.{obj_key}.regions is required when present=true")

        organization = _upper_code(organization_raw)
        topology = _upper_code(topology_raw)
        if organization not in ORGANIZATION_CODES:
            raise ValueError(f"invalid content.{obj_key}.organization: {organization_raw!r}")
        if topology not in TOPOLOGY_CODES:
            raise ValueError(f"invalid content.{obj_key}.topology: {topology_raw!r}")

        if isinstance(regions_raw, str):
            regions_raw = [x for x in regions_raw.replace(";", ",").split(",") if x.strip()]
        if not isinstance(regions_raw, Sequence):
            raise ValueError(f"content.{obj_key}.regions must be a list")
        regions = _unique_codes(regions_raw, REGION_CODES, field=f"content.{obj_key}.regions")
        regions.sort(key=_REGION_ORDER.index)
        if not regions:
            raise ValueError(f"content.{obj_key}.regions must contain at least one reliable R code")

        if morph_raw is None:
            morph_raw = []
        if isinstance(morph_raw, str):
            morph_raw = [x for x in morph_raw.replace(";", ",").split(",") if x.strip()]
        if not isinstance(morph_raw, Sequence):
            raise ValueError(f"content.{obj_key}.morphology must be a list")
        morphology = _unique_codes(morph_raw, MORPHOLOGY_CODES, field=f"content.{obj_key}.morphology")
        
        
        if not morphology:
            morphology = ["M0"]
        morphology.sort(key=_MORPH_ORDER.index)

        content[obj_key] = {
            "present": True,
            "organization": organization,
            "regions": regions,
            "topology": topology,
            "morphology": morphology,
        }

    
    if not enable_train_slot and isinstance(content_map, Mapping):
        train_raw = content_map.get("train")
        if isinstance(train_raw, Mapping) and _parse_bool(train_raw.get("present", False)):
            raise ValueError("train branch is present but train slot is disabled")

    return {"geometry": geometry, "environment": environment, "content": content}






def codebook_bit_cost(codes: Mapping[str, Any], *, enable_train_slot: bool = False) -> int:
    c = canonicalize_codes(codes, enable_train_slot=enable_train_slot)
    base = GEOMETRY_BITS + ENVIRONMENT_BITS + CONTENT_CORE_BITS + (1 if enable_train_slot else 0)
    active = sum(int(c["content"][k]["present"]) for k in active_object_keys(enable_train_slot))
    return base + active * OBJECT_BRANCH_BITS


def _pack_mask(active_codes: Sequence[str], ordered_codes: Sequence[str]) -> int:
    active = set(active_codes)
    value = 0
    for i, code in enumerate(ordered_codes):
        if code in active:
            value |= 1 << i
    return value


def _unpack_mask(value: int, ordered_codes: Sequence[str]) -> List[str]:
    return [code for i, code in enumerate(ordered_codes) if value & (1 << i)]


def pack_codes(codes: Mapping[str, Any], *, enable_train_slot: bool = False) -> int:
    c = canonicalize_codes(codes, enable_train_slot=enable_train_slot)
    value = 0
    shift = 0

    value |= _GEOMETRY_INDEX[c["geometry"]] << shift
    shift += GEOMETRY_BITS

    value |= _pack_mask(c["environment"], _ENV_ORDER) << shift
    shift += ENVIRONMENT_BITS

    obj_keys = active_object_keys(enable_train_slot)
    for i, key in enumerate(obj_keys):
        if c["content"][key]["present"]:
            value |= 1 << (shift + i)
    shift += len(obj_keys)

    for key in obj_keys:
        branch = c["content"][key]
        if not branch["present"]:
            continue
        value |= _ORG_INDEX[branch["organization"]] << shift
        shift += ORGANIZATION_BITS
        value |= _pack_mask(branch["regions"], _REGION_ORDER) << shift
        shift += REGION_BITS
        value |= _TOPOLOGY_INDEX[branch["topology"]] << shift
        shift += TOPOLOGY_BITS
        value |= _pack_mask(branch["morphology"], _MORPH_ORDER) << shift
        shift += MORPHOLOGY_BITS

    return value


def unpack_codes(value: int, *, enable_train_slot: bool = False) -> Dict[str, Any]:
    if not isinstance(value, int) or value < 0:
        raise ValueError("packed value must be a non-negative integer")
    shift = 0
    g_idx = (value >> shift) & ((1 << GEOMETRY_BITS) - 1)
    shift += GEOMETRY_BITS
    if g_idx not in _INDEX_GEOMETRY:
        raise ValueError(f"invalid packed geometry index: {g_idx}")
    geometry = _INDEX_GEOMETRY[g_idx]

    env_mask = (value >> shift) & ((1 << ENVIRONMENT_BITS) - 1)
    shift += ENVIRONMENT_BITS
    environment = _unpack_mask(env_mask, _ENV_ORDER)

    obj_keys = active_object_keys(enable_train_slot)
    c_mask = (value >> shift) & ((1 << len(obj_keys)) - 1)
    shift += len(obj_keys)

    content: Dict[str, Any] = {}
    for i, key in enumerate(obj_keys):
        present = bool(c_mask & (1 << i))
        if not present:
            content[key] = empty_object_branch()
            continue
        o_idx = (value >> shift) & ((1 << ORGANIZATION_BITS) - 1)
        shift += ORGANIZATION_BITS
        if o_idx not in _INDEX_ORG:
            raise ValueError(f"invalid packed organization index {o_idx} for {key}")
        r_mask = (value >> shift) & ((1 << REGION_BITS) - 1)
        shift += REGION_BITS
        regions = _unpack_mask(r_mask, _REGION_ORDER)
        if not regions:
            raise ValueError(f"packed branch {key} has empty regional occupancy")
        t_idx = (value >> shift) & ((1 << TOPOLOGY_BITS) - 1)
        shift += TOPOLOGY_BITS
        if t_idx not in _INDEX_TOPOLOGY:
            raise ValueError(f"invalid packed topology index {t_idx} for {key}")
        m_mask = (value >> shift) & ((1 << MORPHOLOGY_BITS) - 1)
        shift += MORPHOLOGY_BITS
        morphology = _unpack_mask(m_mask, _MORPH_ORDER)


        if not morphology:
            morphology = ["M0"]
        content[key] = {
            "present": True,
            "organization": _INDEX_ORG[o_idx],
            "regions": regions,
            "topology": _INDEX_TOPOLOGY[t_idx],
            "morphology": morphology,
        }


    if value >> shift:
        raise ValueError("packed value contains unexpected trailing high bits")
    return {"geometry": geometry, "environment": environment, "content": content}


def bitstring(codes: Mapping[str, Any], *, enable_train_slot: bool = False) -> str:
    n = codebook_bit_cost(codes, enable_train_slot=enable_train_slot)
    return format(pack_codes(codes, enable_train_slot=enable_train_slot), f"0{n}b")

def semantic_r10_definition_text() -> str:
    """Definition shown ONLY to the second-pass region localizer.

    CoverageFirst/BandPriority rules:
      * first determine the TRUE dominant spatial coverage of reliable target centers;
      * then choose 1, 2, or 3 semantic regions according to that coverage;
      * do not prefer a single label in advance;
      * broad left / middle / right bands are symmetric;
      * center is reserved for a compact central cluster;
      * corner labels are exceptional refinements for obvious compact corner clusters.

    The task is semantic summarization of salient target locations, NOT exhaustive
    occupancy enumeration.
    """
    lines = [
        "R — Semantic Dominant Region Summary (choose 1-3 labels according to true spatial coverage):",
    ]
    for code, info in REGION_CODES.items():
        lines.append(f"- {code}: {info['en']}")
    lines.extend([
        "R selection rules (IMPORTANT):",
        "1. Localize reliable salient target OBJECT CENTERS only. Ignore roads, background structures, and weak/uncertain responses.",
        "2. This is semantic spatial summarization, NOT exhaustive grid occupancy. Do NOT list every small location touched by a target.",
        "3. Do NOT prefer one, two, or three labels in advance. FIRST determine the true dominant spatial coverage, THEN choose 1-3 labels accordingly.",
        "4. Choose the MINIMUM number of labels that FAITHFULLY PRESERVES the dominant target distribution. Do not discard a clearly occupied second or third major region merely because another region contains more targets.",
        "5. Treat the three vertical macro bands SYMMETRICALLY: R5=left-side, R2=middle, R4=right-side.",
        "   - targets distributed along the LEFT side from top to bottom, even if sparse/band-like -> [R5]",
        "   - targets distributed through the MIDDLE vertical band / central axis from top to bottom -> [R2]",
        "   - targets distributed along the RIGHT side from top to bottom, even if sparse/band-like -> [R4]",
        "6. Horizontal macro bands: a broad distribution mainly across the UPPER horizontal band -> [R1]; mainly across the LOWER horizontal band -> [R3].",
        "7. Center rule: use R6 ONLY for a clearly COMPACT cluster around the image center. A long/scattered middle-axis distribution crossing the center is R2, NOT R6.",
        "8. Corner rule: R7/R8/R9/R10 are EXCEPTIONAL refinement labels. Use a corner label only when a compact salient target cluster is clearly and predominantly confined to that corner.",
        "   - compact upper-left cluster -> [R7]",
        "   - compact lower-left cluster -> [R8]",
        "   - compact upper-right cluster -> [R9]",
        "   - compact lower-right cluster -> [R10]",
        "   - never use a corner label merely because scattered targets happen to touch that corner.",
        "9. Multi-region coverage rule:",
        "   - one genuinely dominant semantic region -> 1 label",
        "   - two clearly occupied dominant semantic regions -> 2 labels",
        "   - broad distribution spanning left + middle + right -> [R5,R2,R4]",
        "   - middle + right coverage -> [R2,R4]",
        "   - left + middle coverage -> [R5,R2]",
        "   - do NOT collapse clearly occupied multiple macro bands into one label just because one band has the largest count.",
        "10. Examples:",
        "   - scattered targets along the whole right side -> [R4]",
        "   - targets distributed from top to bottom through the middle -> [R2]",
        "   - compact group around the image center -> [R6]",
        "   - compact group only in lower-left -> [R8]",
        "   - targets spanning middle and right bands -> [R2,R4]",
        "   - targets broadly spanning the image width -> [R5,R2,R4]",
        "11. NEVER output more than 3 labels. NEVER enumerate all corner labels. NEVER output R1-R10 together.",
        "12. If coverage is ambiguous, preserve every CLEARLY OCCUPIED MAJOR MACRO BAND up to 3 labels. Do NOT force a single-label answer and do NOT use a majority-only fallback.",
        "13. Do not infer or modify topology, density, morphology, object presence, geometry, or environment in this pass.",
    ])
    return "\n".join(lines)


def canonicalize_region_selection(selection: Mapping[str, Any], *, active_keys: Sequence[str]) -> Dict[str, List[str]]:
    """Validate second-pass Semantic-R10 selections for fixed active families."""
    if isinstance(selection.get("regions"), Mapping):
        selection = selection["regions"]  
    if not isinstance(selection, Mapping):
        raise ValueError("region-only output must be an object keyed by active object family")
    out: Dict[str, List[str]] = {}
    expected = list(active_keys)
    for key in expected:
        raw = selection.get(key)
        if raw is None:
            raise ValueError(f"missing Semantic-R10 regions for active family: {key}")
        if isinstance(raw, str):
            raw = [x for x in raw.replace(";", ",").split(",") if x.strip()]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError(f"regions.{key} must be a JSON list")
        vals = _unique_codes(raw, REGION_CODES, field=f"regions.{key}")
        vals.sort(key=_REGION_ORDER.index)
        if not (1 <= len(vals) <= 3):
            raise ValueError(f"regions.{key} must contain 1-3 Semantic-R10 codes; got {vals}")
        out[key] = vals
    extras = [k for k in selection.keys() if k not in expected]
    if extras:
        raise ValueError(f"region-only output contains unexpected/inactive families: {extras}")
    return out


def merge_regions_only(
    base_codes: Mapping[str, Any],
    region_selection: Mapping[str, Any],
    *,
    enable_train_slot: bool = False,
) -> Dict[str, Any]:
    base = canonicalize_codes(base_codes, enable_train_slot=enable_train_slot)
    active = [k for k in active_object_keys(enable_train_slot) if base["content"][k]["present"]]
    selected = canonicalize_region_selection(region_selection, active_keys=active)
    merged = deepcopy(base)
    for key in active:
        merged["content"][key]["regions"] = list(selected[key])
    merged = canonicalize_codes(merged, enable_train_slot=enable_train_slot)

    
    if merged["geometry"] != base["geometry"] or merged["environment"] != base["environment"]:
        raise RuntimeError("region-only merge modified G/E unexpectedly")
    for key in active_object_keys(enable_train_slot):
        b0, b1 = base["content"][key], merged["content"][key]
        for field in ("present", "organization", "topology", "morphology"):
            if b0.get(field) != b1.get(field):
                raise RuntimeError(f"region-only merge modified non-R field: {key}.{field}")
    return merged



def exact_json_template(*, enable_train_slot: bool = False) -> str:
    branches = [
        '"human":{"present":true,"organization":"O?","regions":["R?"],"topology":"T?","morphology":["M?"]}',
        '"vehicle":{"present":true,"organization":"O?","regions":["R?"],"topology":"T?","morphology":["M?"]}',
    ]
    if enable_train_slot:
        branches.append('"train":{"present":false}')
    return '{"geometry":"G?","environment":["E_?"],"content":{' + ",".join(branches) + '}}'


def _join_phrases(items: Sequence[str]) -> str:
    items = [x for x in items if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _region_support_phrase(regions: Sequence[str]) -> str:
    """Render only the final Semantic-R10 region field.

    No topology or organization semantics are inferred here; those fields are
    frozen from the legacy first pass.
    """
    r = tuple(sorted(regions, key=_REGION_ORDER.index))
    names = [REGION_CODES[x]["render"] for x in r]
    if len(names) == 1:
        return f"in the {names[0]} region"
    return "mainly across the " + _join_phrases(names) + " regions"


def _morph_phrase(morphology: Sequence[str], modality: str) -> str:
    if not morphology:
        return ""
    vals = [MORPHOLOGY_CODES[m][modality] for m in morphology]
    return _join_phrases(vals)


def _object_phrase(obj_key: str, branch: Mapping[str, Any], *, modality: str) -> str:
    noun_rgb = {"human": "human silhouettes", "vehicle": "vehicle structures", "train": "large transport structures"}
    noun_ir = {"human": "human thermal responses", "vehicle": "vehicle thermal responses", "train": "large transport thermal responses"}
    noun = (noun_rgb if modality == "rgb" else noun_ir)[obj_key]
    org = ORGANIZATION_CODES[branch["organization"]]["render"]
    region = _region_support_phrase(branch["regions"])
    topology = TOPOLOGY_CODES[branch["topology"]]["render"]
    morph = _morph_phrase(branch.get("morphology", []), modality)
    text = f"Preserve {org} {noun} {region}, {topology}"
    if morph:
        text += f", with {morph}"
    return text + "."


def render_tsp_prompts(codes: Mapping[str, Any], *, enable_train_slot: bool = False) -> Dict[str, str]:
    c = canonicalize_codes(codes, enable_train_slot=enable_train_slot)
    geom = GEOMETRY_CODES[c["geometry"]]["render"]
    env_rgb = _join_phrases([ENVIRONMENT_CODES[e]["rgb"] for e in c["environment"]])
    env_ir = _join_phrases([ENVIRONMENT_CODES[e]["ir"] for e in c["environment"]])

    rgb_lines = [f"Thermal side information indicates a {geom} scene structure."]
    ir_lines = [f"Preserve the {geom} thermal scene structure."]
    if env_rgb:
        rgb_lines.append(f"Preserve the coarse scene context containing {env_rgb}.")
    if env_ir:
        ir_lines.append(f"Preserve coarse thermal context associated with {env_ir}.")

    for key in active_object_keys(enable_train_slot):
        branch = c["content"][key]
        if branch["present"]:
            rgb_lines.append(_object_phrase(key, branch, modality="rgb"))
            ir_lines.append(_object_phrase(key, branch, modality="ir"))

    if not any(c["content"][k]["present"] for k in active_object_keys(enable_train_slot)):
        rgb_lines.append("No reliable salient human- or vehicle-like thermal object branch is declared.")
        ir_lines.append("No reliable salient human- or vehicle-like thermal object branch is declared.")

    rgb_lines.append("Use these IR-derived structured cues as auxiliary guidance for RGB compression decoding.")
    shared = (
        f"Structured TSP code: G={c['geometry']}; E={','.join(c['environment']) or 'none'}; "
        + "; ".join(
            f"{key}=" + (
                f"{b['organization']},R={','.join(b['regions'])},{b['topology']},M={','.join(b['morphology']) or 'none'}"
                if b["present"] else "absent"
            )
            for key, b in c["content"].items()
        )
        + "."
    )
    return {"rgb": " ".join(rgb_lines), "ir": " ".join(ir_lines), "shared": shared}




def codes_to_nested_json(codes: Mapping[str, Any], *, enable_train_slot: bool = False) -> Dict[str, Any]:
    return {"thermal_side_prompt": canonicalize_codes(codes, enable_train_slot=enable_train_slot)}


def flatten_nested_json(obj: Mapping[str, Any], *, enable_train_slot: bool = False) -> Dict[str, Any]:
    """Compatibility name retained from v1; returns canonical v3 structured codes."""
    return canonicalize_codes(obj, enable_train_slot=enable_train_slot)




if __name__ == "__main__":
    example = {
        "geometry": "G1",
        "environment": ["E_P", "E_V"],
        "content": {
            "human": {"present": True, "organization": "O2", "regions": ["R1", "R5"], "topology": "T3", "morphology": ["M1"]},
            "vehicle": {"present": True, "organization": "O3", "regions": ["R6"], "topology": "T2", "morphology": ["M2"]},
        },
    }
    packed = pack_codes(example)
    print(VERSION)
    print("bit_cost=", codebook_bit_cost(example), "packed=", packed, "bitstring=", bitstring(example))
    print(unpack_codes(packed))
    print(render_tsp_prompts(example)["rgb"])
