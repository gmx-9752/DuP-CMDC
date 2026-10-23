#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict as edict

from lib.diffc.encode import encode
from lib.diffc.rcc.gaussian_channel_simulator import GaussianChannelSimulator
from lib.models.SanaDiT_dup_cmdc import SanaDiTModel_DuPCmdc
from lib.dup_cmdc.image_io import (
    load_rgb_tensor,
    load_thermal_tensor,
)
from lib.dup_cmdc.pairing import (
    build_paired_items,
    pair_key_from_path,
)
from lib.dup_cmdc.text_cache import TriTSPTextCache
from lib.dup_cmdc.tsp_core import (
    load_tsp_core_json,
    iter_tsp_primitive_specs,
    normalize_tsp_core,
    read_path_list,
)
from lib.dup_cmdc.tsp_schema import (
    GEOMETRY_CODES, ENVIRONMENT_CODES, ORGANIZATION_CODES,
    REGION_CODES, TOPOLOGY_CODES, MORPHOLOGY_CODES,
)


MAGIC = b'TRITSP03'
VERSION = 1
HEADER = struct.Struct('<8sBBBBHHH8H4I16s16s16s')
MODES = ('full',)
PREPROCESS = ('pad64', 'resize', 'none')
MODE_ID = {k: i for i, k in enumerate(MODES)}
PREPROCESS_ID = {k: i for i, k in enumerate(PREPROCESS)}
RGB_FLAG, THERMAL_FLAG, TSP_FLAG = 1, 2, 4
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', '--ckpt', dest='checkpoint', required=True)
    p.add_argument('--text_cache', default=None, help='Defaults to text_cache.pt next to --checkpoint')
    p.add_argument('--output_dir', default='./outputs/compressed')
    p.add_argument('--recon_timesteps', '--recon_timestep', dest='recon_timesteps', required=True, help='Comma-separated STOP timesteps in config.encoding_timesteps')
    source = p.add_argument_group('input sources (choose ONE shape)')
    source.add_argument('--rgb_list')
    source.add_argument('--thermal_list', '--ir_list', dest='thermal_list')
    source.add_argument('--prompt_list', help='JSON path list, paired by canonical scene/filename key')
    source.add_argument('--rgb_path')
    source.add_argument('--thermal_path', '--ir_path', dest='thermal_path')
    source.add_argument('--prompt_path', help='One TSP JSON for the selected pair/image')
    source.add_argument('--image_path', help='Legacy single-image input (use --modality)')
    source.add_argument('--image_dir', help='Legacy single-modality folder (use --modality)')
    source.add_argument('--modality', choices=['rgb', 'thermal', 'ir'], default='rgb')
    source.add_argument('--recursive', action='store_true')
    p.add_argument('--preprocess_mode', choices=PREPROCESS, default='pad64')
    p.add_argument('--target_width', type=int, default=1024)
    p.add_argument('--target_height', type=int, default=576)
    p.add_argument('--compression_seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--high_boundary', type=int, default=300)
    p.add_argument('--middle_boundary', type=int, default=85)
    p.add_argument('--tritsp_hidden_size', type=int, default=None)
    p.add_argument('--tritsp_cond_dim', type=int, default=512)
    p.add_argument('--tritsp_num_heads', type=int, default=8)
    p.add_argument('--tritsp_condition_tokens', type=int, default=12)
    p.add_argument('--tritsp_tsp_condition_tokens', type=int, default=8)
    p.add_argument('--tritsp_spp_condition_tokens', type=int, default=4)
    p.add_argument('--tritsp_condition_decoder_layers', type=int, default=2)
    p.add_argument('--tritsp_delta_rank', type=int, default=64)
    p.add_argument('--tritsp_gate_max', type=float, default=0.5)
    p.add_argument('--tritsp_inject_block_indices', default='9,14,18,23')
    return p.parse_args()


def canonical_mode(saved_ckpt: Mapping[str, Any]) -> str:
    extra = saved_ckpt.get("extra") or {}
    saved_mode = str(extra.get("train_condition_mode", (extra.get("args") or {}).get("train_condition_mode", "full"))).lower()
    if saved_mode != "full":
        raise ValueError("Checkpoint was not trained in full mode")
    if saved_ckpt.get("architecture_version") not in (
        "dup_cmdc_full_v1", "xiaorong_split_prior_component_isolation_v3",
        "xiaorong_split_prior_decoupled_v2"
    ):
        raise ValueError("Unsupported checkpoint architecture")
    return "full"


def source_items(args, needs_tsp=True):
    using_lists = bool(args.rgb_list or args.thermal_list)
    using_pair = bool(args.rgb_path or args.thermal_path)
    using_single = bool(args.image_path or args.image_dir)
    if sum((using_lists, using_pair, using_single)) != 1:
        raise ValueError('Provide paired lists, a paired RGB/Thermal image, or a single-modality input')
    if using_lists:
        if not args.rgb_list or not args.thermal_list or not args.prompt_list:
            raise ValueError('Paired lists require --rgb_list, --thermal_list and --prompt_list')
        return [
            (key, rgb, thermal, codes)
            for rgb, thermal, _prompt, codes, key in build_paired_items(
                args.rgb_list, args.thermal_list, args.prompt_list
            )
        ]
    if using_pair:
        if not args.rgb_path or not args.thermal_path or not args.prompt_path:
            raise ValueError('Paired input requires --rgb_path, --thermal_path and --prompt_path')
        rgb, thermal = Path(args.rgb_path), Path(args.thermal_path)
        if not rgb.is_file() or not thermal.is_file():
            raise FileNotFoundError(f'Input not found: {rgb} / {thermal}')
        key1 = pair_key_from_path(rgb, 'rgb')
        key2 = pair_key_from_path(thermal, 'thermal')
        if key1 != key2:
            raise ValueError(f'RGB/Thermal pair mismatch: {key1} vs {key2}')
        return [(key1, rgb, thermal, require_codes(args.prompt_path))]
    if bool(args.image_path) == bool(args.image_dir):
        raise ValueError('Specify exactly one of --image_path or --image_dir')
    modality = 'thermal' if args.modality in ('thermal', 'ir') else 'rgb'
    if args.image_path:
        paths = [Path(args.image_path)]
    else:
        directory = Path(args.image_dir)
        paths = sorted(directory.rglob('*') if args.recursive else directory.glob('*'))
        paths = [p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not paths:
        raise ValueError('No input images found')
    prompt_map = {}
    if args.prompt_list:
        for path in read_path_list(args.prompt_list):
            key = pair_key_from_path(path, 'prompt')
            if key in prompt_map:
                raise ValueError(f'Duplicate TSP prompt key: {key}')
            prompt_map[key] = path
    elif len(paths) != 1 or not args.prompt_path:
        raise ValueError('A folder requires --prompt_list; a single image requires --prompt_path')
    result = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        key = pair_key_from_path(path, modality)
        if prompt_map and key not in prompt_map:
            raise KeyError(f'Missing TSP prompt for {key}')
        codes = require_codes(prompt_map[key] if prompt_map else args.prompt_path)
        result.append((key, path if modality == 'rgb' else None,
                       path if modality == 'thermal' else None, codes))
    return result


def require_codes(path):
    if not path:
        raise ValueError('Missing TSP JSON for full mode')
    return load_tsp_core_json(path)


def parse_recon_timesteps(text, schedule):
    values = [int(x) for x in str(text).replace(';', ',').replace(' ', ',').split(',') if x.strip()]
    if not values:
        raise ValueError('Provide at least one recon_timestep')
    missing = [t for t in values if t not in schedule]
    if missing:
        raise ValueError(f'Recon timesteps not in config.encoding_timesteps: {missing}')
    return sorted(set(values), reverse=True)


def get_config(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = edict(yaml.safe_load(f))
    schedule = [int(t) for t in cfg.encoding_timesteps]
    if len(schedule) != len(set(schedule)) or any(schedule[i] <= schedule[i + 1] for i in range(len(schedule) - 1)):
        raise ValueError('encoding_timesteps must be strictly decreasing and unique')
    if args.recon_timestep not in schedule:
        raise ValueError(f'Stop timestep {args.recon_timestep} not found in encoding_timesteps={schedule}')
    manual = cfg.get('manual_dkl_per_step')
    if manual is None:
        raise ValueError('Self-contained DiffC decoding requires manual_dkl_per_step in YAML. '
                         'The current variable-DKL encoder cannot reconstruct its decoder without DKL side information.')
    if len(manual) != len(schedule):
        raise ValueError('manual_dkl_per_step must have the same length as encoding_timesteps')
    stop_idx = schedule.index(args.recon_timestep)
    cfg.encoding_timesteps = schedule[:stop_idx + 1]
    cfg.manual_dkl_per_step = list(manual[:stop_idx + 1])
    if args.tritsp_tsp_condition_tokens + args.tritsp_spp_condition_tokens != args.tritsp_condition_tokens:
        raise ValueError('TSP queries + SPP queries must equal all condition queries')
    if not 0 <= args.compression_seed <= 0xFFFFFFFF:
        raise ValueError('--compression_seed must fit uint32')
    return cfg, stop_idx


def relevant_config(config, args, mode):
    """Canonical, path-independent algorithm identity, shared with decompress.py."""
    return {
        'version': VERSION, 'mode': mode,
        'encoding_timesteps': list(config.encoding_timesteps),
        'manual_dkl_per_step': list(config.manual_dkl_per_step),
        'denoising_timesteps': list(config.denoising_timesteps),
        'max_chunk_size': config.max_chunk_size, 'chunk_padding': config.chunk_padding,
        'encoding_guidance_scale': config.encoding_guidance_scale,
        'denoising_guidance_scale': config.denoising_guidance_scale,
        'model_path': config.get('model_path', config.get('model_id')),
        'transformer_dtype': config.get('transformer_dtype', 'float16'),
        'text_vae_dtype': config.get('text_vae_dtype', 'bfloat16'),
        'max_sequence_length': config.get('max_sequence_length', 300),
        'preprocess_mode': args.preprocess_mode,
        'target_width': args.target_width, 'target_height': args.target_height,
        'high_boundary': args.high_boundary, 'middle_boundary': args.middle_boundary,
        'hidden_size': args.tritsp_hidden_size, 'cond_dim': args.tritsp_cond_dim,
        'num_heads': args.tritsp_num_heads,
        'condition_tokens': args.tritsp_condition_tokens,
        'tsp_tokens': args.tritsp_tsp_condition_tokens,
        'spp_tokens': args.tritsp_spp_condition_tokens,
        'decoder_layers': args.tritsp_condition_decoder_layers,
        'delta_rank': args.tritsp_delta_rank, 'gate_max': args.tritsp_gate_max,
        'inject_blocks': args.tritsp_inject_block_indices,
    }


def digest_config(config, args, mode):
    raw = json.dumps(relevant_config(config, args, mode), sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(raw.encode('utf-8')).digest()[:16]


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.digest()[:16]


def pack_tsp(codes):
    """Lossless sparse packing. 10 base bits + 18 bits per PRESENT family.

    The supplied schema declares MORPHOLOGY_BITS=3 for FOUR codes M0-M3.
    A three-bit mask cannot encode all four independently. We intentionally
    allocate FOUR morphology bits here instead of silently dropping M3.
    This corrects the on-wire representation without changing the model.
    """
    c = normalize_tsp_core(codes)
    g, env, org, region, topo, morph = [list(v) for v in (
        GEOMETRY_CODES, ENVIRONMENT_CODES, ORGANIZATION_CODES,
        REGION_CODES, TOPOLOGY_CODES, MORPHOLOGY_CODES)]
    x, shift = 0, 0
    def put(value, bits):
        nonlocal x, shift
        if value < 0 or value >= (1 << bits):
            raise ValueError(f'TSP value {value} does not fit {bits} bits')
        x |= value << shift
        shift += bits
    def mask(values, order):
        return sum(1 << order.index(v) for v in values)
    put(g.index(c['geometry']), 3)
    put(mask(c['environment'], env), 5)
    active = [c['content'][k]['present'] for k in ('human', 'vehicle')]
    put(int(active[0]) + 2 * int(active[1]), 2)
    for key, present in zip(('human', 'vehicle'), active):
        if present:
            branch = c['content'][key]
            put(org.index(branch['organization']), 2)
            put(mask(branch['regions'], region), 10)
            put(topo.index(branch['topology']), 2)
            put(mask(branch['morphology'], morph), 4)
    binary = x.to_bytes((shift + 7) // 8, 'little')
    if unpack_tsp(binary) != c:
        raise RuntimeError('TSP codebook packing failed round-trip test')
    return binary


def unpack_tsp(binary):
    """Decoder-side counterpart, used for encoder round-trip validation."""
    if len(binary) not in (2, 4, 6):
        raise ValueError(f'Unexpected TSP payload length {len(binary)}')
    g, env, org, region, topo, morph = [list(v) for v in (
        GEOMETRY_CODES, ENVIRONMENT_CODES, ORGANIZATION_CODES,
        REGION_CODES, TOPOLOGY_CODES, MORPHOLOGY_CODES)]
    value, offset = int.from_bytes(binary, 'little'), 0
    def get(bits):
        nonlocal offset
        field = (value >> offset) & ((1 << bits) - 1)
        offset += bits
        return field
    def get_mask(bits, order):
        raw = get(bits)
        return [v for i, v in enumerate(order) if raw & (1 << i)]
    geom_idx = get(3)
    if geom_idx >= len(g):
        raise ValueError('Invalid geometry code index')
    code = {'geometry': g[geom_idx], 'environment': get_mask(5, env), 'content': {}}
    active = get(2)
    for i, name in enumerate(('human', 'vehicle')):
        if not (active & (1 << i)):
            code['content'][name] = {'present': False}
            continue
        org_idx = get(2)
        regs = get_mask(10, region)
        topo_idx = get(2)
        morphs = get_mask(4, morph)
        if org_idx >= len(org) or topo_idx >= len(topo) or not regs or not morphs:
            raise ValueError('Invalid sparse object branch in TSP payload')
        code['content'][name] = {
            'present': True, 'organization': org[org_idx],
            'regions': regs, 'topology': topo[topo_idx], 'morphology': morphs,
        }
    if value >> offset:
        raise ValueError('Non-zero trailing bits in TSP payload')
    return normalize_tsp_core(code)


def build_model(config, args):
    return SanaDiTModel_DuPCmdc(
        model_id=config.get('model_path', config.get('model_id')),
        device=args.device,
        transformer_dtype=config.get('transformer_dtype', 'float16'),
        text_vae_dtype=config.get('text_vae_dtype', 'bfloat16'),
        local_files_only=config.get('local_files_only', True),
        max_sequence_length=config.get('max_sequence_length', 300),
        tritsp_ckpt=args.checkpoint,
        tritsp_text_cache_path=args.text_cache,
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


def benchmark_parameters_dup(model, mode):
    """Frozen SANA pipeline + trained adapter; encoder-side online Qwen excluded."""
    base = getattr(model, 'pipe', None)
    components = getattr(base, 'components', {}) if base is not None else {}
    modules = [m for m in components.values() if isinstance(m, torch.nn.Module)]
    if not modules:
        modules = [getattr(model, key) for key in ('transformer', 'vae')
                   if isinstance(getattr(model, key, None), torch.nn.Module)]
        text_encoder = getattr(base, 'text_encoder', None)
        if isinstance(text_encoder, torch.nn.Module):
            modules.append(text_encoder)
    modules.append(model.tritsp_adapter)
    all_params = {id(p): p for module in modules for p in module.parameters()}
    
    train_params = model.trainable_parameters(mode)
    train_params = {id(p): p for p in train_params}
    return {'total_parameters': sum(p.numel() for p in all_params.values()),
            'trainable_parameters': sum(p.numel() for p in train_params.values()),
            'trainable_definition': 'DuP-CMDC optimizer-owned adapter parameters (training condition mode)',
            'total_scope': 'frozen SANA + VAE + text encoder + adapter; excludes external Qwen TSP generator'}


def write_benchmark_summary_dup(path, rows, params, keys):
    import statistics
    summary = dict(num_records=len(rows), **params,
                   total_params_m=params['total_parameters']/1e6,
                   trainable_params_m=params['trainable_parameters']/1e6)
    for key in keys:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        summary['average_'+key] = statistics.mean(values) if values else None
        summary['std_'+key] = statistics.stdev(values) if len(values)>1 else 0.0
        summary['count_'+key] = len(values)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary

def gpu_sync(device):
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def encode_modality(path, modality, codes, model, simulator, config, args, mode, seed):
    if path is None:
        return b'', (0, 0, 0, 0), None
    if modality == 'rgb':
        x, raw_hw, _ = load_rgb_tensor(
            path, args.preprocess_mode, args.target_width, args.target_height)
    else:
        x, _, raw_hw, _ = load_thermal_tensor(
            path, args.preprocess_mode, args.target_width, args.target_height)
    raw_h, raw_w = map(int, raw_hw)
    proc_h, proc_w = map(int, x.shape[-2:])
    shape = (raw_w, raw_h, proc_w, proc_h)
    if any(v <= 0 or v > 65535 for v in shape):
        raise ValueError(f'Unsupported image dimensions {shape}, must fit uint16')
    with torch.inference_mode():
        gpu_sync(args.device)
        start = time.perf_counter()
        model.set_condition(codes, modality, enabled=True, condition_mode=mode)
        model.configure_empty(config.encoding_guidance_scale, proc_w, proc_h)
        gt_latent = model.image_to_latent(x.to(model.device, dtype=torch.float32))
        chunk_seeds, dkl_steps, _, indices = encode(
            gt_latent, config.encoding_timesteps, model, simulator,
            config.manual_dkl_per_step, [int(args.recon_timestep)], seed=seed)
        if len(indices) != 1 or indices[0] != len(config.encoding_timesteps) - 1:
            raise RuntimeError(f'Unexpected stop step indices from DiffC: {indices}')
        bitstream = bytes(simulator.compress_chunk_seeds(chunk_seeds, dkl_steps))
        gpu_sync(args.device)
        seconds = time.perf_counter() - start
    return bitstream, shape, seconds


def save_container(path, *, mode, preprocess, stop_idx, timestep, rgb_shape, thermal_shape,
                   rgb_seed, thermal_seed, rgb_stream, thermal_stream, tsp_binary,
                   config_hash, ckpt_hash, cache_hash):
    flags = (RGB_FLAG if rgb_stream is not None else 0) | (THERMAL_FLAG if thermal_stream is not None else 0)
    flags |= (TSP_FLAG if tsp_binary else 0)
    r = rgb_stream if rgb_stream is not None else b''
    t = thermal_stream if thermal_stream is not None else b''
    if not flags & (RGB_FLAG | THERMAL_FLAG):
        raise ValueError('At least one modality must be encoded')
    if any(x < 0 or x > 65535 for x in (stop_idx, timestep, len(tsp_binary))):
        raise ValueError('Invalid timestep/header field')
    packed = HEADER.pack(
        MAGIC, VERSION, MODE_ID[mode], PREPROCESS_ID[preprocess], flags,
        stop_idx, timestep, len(tsp_binary),
        *(rgb_shape + thermal_shape), int(rgb_seed), int(thermal_seed), len(r), len(t),
        config_hash, ckpt_hash, cache_hash,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.partial')
    try:
        with open(tmp, 'wb') as f:
            f.write(packed)
            f.write(tsp_binary)
            f.write(r)
            f.write(t)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return target.stat().st_size



def setup_parallel(args):
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if str(args.device).startswith('cuda'):
        device_index = local_rank if world_size > 1 else (
            int(args.device.split(':', 1)[1]) if ':' in args.device else 0
        )
        if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
            raise RuntimeError(f'CUDA device {device_index} unavailable for local rank {local_rank}')
        torch.cuda.set_device(device_index)
        args.device = f'cuda:{device_index}'
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError('torch.distributed is required for multiple workers')
        dist.init_process_group(backend='gloo', init_method='env://')
    return rank, world_size


def collect_rows(rows, world_size):
    if world_size == 1:
        return rows
    collected = [None] * world_size
    dist.all_gather_object(collected, rows)
    return [row for subset in collected for row in subset]


def main():
    args = parse_args()
    if args.text_cache is None:
        args.text_cache = str(Path(args.checkpoint).parent / 'text_cache.pt')
    rank, world_size = setup_parallel(args)
    checkpoint_path, cache_path = Path(args.checkpoint), Path(args.text_cache)
    for f in (checkpoint_path, cache_path):
        if not f.is_file():
            raise FileNotFoundError(f)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    mode = canonical_mode(checkpoint)
    del checkpoint
    entries = source_items(args, needs_tsp=True)
    if len({item[0] for item in entries}) != len(entries):
        raise ValueError('Duplicate output canonical keys')
    with open(args.config, 'r', encoding='utf-8') as f:
        full_config = edict(yaml.safe_load(f))
    recon_timesteps = parse_recon_timesteps(args.recon_timesteps, list(map(int, full_config.encoding_timesteps)))
    args.recon_timestep = min(recon_timesteps)
    model_config, _ = get_config(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_hash = digest_file(checkpoint_path)
    cache_hash = digest_file(cache_path)
    local_entries = [(i, item) for i, item in enumerate(entries) if i % world_size == rank]
    model = None
    simulator = None
    if local_entries:
        model = build_model(model_config, args)
        model.tritsp_text_cache = TriTSPTextCache.load(cache_path)
        for _pair_idx, (_key, _rgb, _thermal, codes) in local_entries:
            for modality in ('rgb', 'thermal'):
                for token_key, _, _, _ in iter_tsp_primitive_specs(codes):
                    cache_key = model.tritsp_text_cache.primitive_key(token_key, modality)
                    if cache_key not in model.tritsp_text_cache.entries:
                        raise KeyError(f'Text cache missing {cache_key}; use matching evaluation cache')
        for stage in ('high', 'middle', 'low'):
            for modality in ('rgb', 'thermal'):
                if model.tritsp_text_cache.stage_key(stage, modality) not in model.tritsp_text_cache.entries:
                    raise KeyError('Text cache is missing stage entries; use the evaluation text cache')
        model.prepare_base_prompt_cache([
            model_config.encoding_guidance_scale, model_config.denoising_guidance_scale
        ])
        model.offload_text_encoder()
        model.tritsp_adapter.eval()
        simulator = GaussianChannelSimulator(model_config.max_chunk_size, model_config.chunk_padding)
    manifest_rows = []
    multiple_timesteps = len(recon_timesteps) > 1
    fields = (
        'average_prompt_bpp', 'average_rgb_bpp', 'average_thermal_bpp',
        'average_rgb_total_bpp', 'average_ir_total_bpp'
    )
    for timestep in recon_timesteps:
        args.recon_timestep = timestep
        config, stop_idx = get_config(args)
        config_hash = digest_config(config, args, mode)
        timestep_dir = out / str(timestep).zfill(3) if multiple_timesteps else out
        timestep_dir.mkdir(parents=True, exist_ok=True)
        for pair_idx, (key, rgb_path, thermal_path, codes) in local_entries:
            seed = args.compression_seed + pair_idx
            if seed > 0xFFFFFFFF:
                raise ValueError('compression_seed + pair_index exceeds uint32')
            tsp_binary = pack_tsp(codes)
            r, r_shape, _ = encode_modality(rgb_path, 'rgb', codes, model, simulator, config, args, mode, seed)
            t, t_shape, _ = encode_modality(thermal_path, 'thermal', codes, model, simulator, config, args, mode, seed)
            archive_path = timestep_dir / (key + '.tritsp')
            size = save_container(
                archive_path, mode=mode, preprocess=args.preprocess_mode,
                stop_idx=stop_idx, timestep=timestep,
                rgb_shape=r_shape, thermal_shape=t_shape, rgb_seed=seed, thermal_seed=seed,
                rgb_stream=r if rgb_path else None, thermal_stream=t if thermal_path else None,
                tsp_binary=tsp_binary, config_hash=config_hash, ckpt_hash=ckpt_hash,
                cache_hash=cache_hash
            )
            if size != HEADER.size + len(tsp_binary) + len(r) + len(t):
                raise RuntimeError(f'Unexpected compressed file size: {archive_path}')
            rgb_pixels = r_shape[0] * r_shape[1] if rgb_path else 0
            thermal_pixels = t_shape[0] * t_shape[1] if thermal_path else 0
            prompt_bits = len(tsp_binary) * 8
            rgb_bits, thermal_bits = len(r) * 8, len(t) * 8
            present = int(bool(rgb_path)) + int(bool(thermal_path))
            allocation = 1.0 / present
            manifest_rows.append({
                '_pair_index': pair_idx,
                'pair_key': key,
                'recon_timestep': timestep,
                'file': str(archive_path),
                'average_prompt_bpp': prompt_bits / (rgb_pixels or thermal_pixels),
                'average_rgb_bpp': rgb_bits / rgb_pixels if rgb_pixels else None,
                'average_thermal_bpp': thermal_bits / thermal_pixels if thermal_pixels else None,
                'average_rgb_total_bpp': (
                    (rgb_bits + allocation * (prompt_bits + HEADER.size * 8)) / rgb_pixels
                    if rgb_pixels else None
                ),
                'average_ir_total_bpp': (
                    (thermal_bits + allocation * (prompt_bits + HEADER.size * 8)) / thermal_pixels
                    if thermal_pixels else None
                ),
            })
    manifest_rows = collect_rows(manifest_rows, world_size)
    if rank == 0:
        timestep_order = {t: i for i, t in enumerate(recon_timesteps)}
        manifest_rows.sort(key=lambda row: (timestep_order[row['recon_timestep']], row['_pair_index']))
        for row in manifest_rows:
            del row['_pair_index']
        summary_rows = []
        for timestep in recon_timesteps:
            rows = [row for row in manifest_rows if row['recon_timestep'] == timestep]
            summary = {'recon_timestep': timestep}
            for field in fields:
                values = [row[field] for row in rows if row[field] is not None]
                summary[field] = sum(values) / len(values) if values else float('nan')
            summary_rows.append(summary)
        with open(out / 'compression_manifest.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['pair_key', 'recon_timestep', 'file', *fields])
            writer.writeheader()
            writer.writerows(manifest_rows)
        with open(out / 'summary_by_timestep.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['recon_timestep', *fields])
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({'recon_timestep': row['recon_timestep'],
                                 **{name: f'{row[name]:.8f}' for name in fields}})
        print('========== Compress BPP Summary ==========')
        for row in summary_rows:
            print(f"\nrecon_timestep = {row['recon_timestep']}")
            for field in fields:
                print(f'{field:<25} = {row[field]:.8f}')
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
