#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict as edict
from PIL import Image

from lib import image_utils
from lib.diffc.decode import decode
from lib.diffc.rcc.gaussian_channel_simulator import GaussianChannelSimulator
from lib.dup_cmdc.image_io import crop_to_raw, gray3
from lib.dup_cmdc.text_cache import TriTSPTextCache


from compress import (
    HEADER, MAGIC, VERSION, MODES, PREPROCESS,
    RGB_FLAG, THERMAL_FLAG, TSP_FLAG,
    build_model, canonical_mode, digest_config, digest_file,
    gpu_sync, unpack_tsp, benchmark_parameters_dup, write_benchmark_summary_dup,
    setup_parallel, collect_rows,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', '--ckpt', dest='checkpoint', required=True)
    p.add_argument('--text_cache', default=None)
    p.add_argument('--input_path', help='Single *.tritsp file')
    p.add_argument('--input_dir', help='Directory with *.tritsp files')
    p.add_argument('--output_dir', default='./outputs/reconstructed')
    p.add_argument('--recursive', action='store_true')
    p.add_argument('--ignore_fingerprints', action='store_true', help='UNSAFE diagnostic: bypass config/weights/cache fingerprint checks')
    p.add_argument('--target_width', type=int, default=1024, help='Match encoder, relevant when --preprocess_mode=resize')
    p.add_argument('--target_height', type=int, default=576)
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


def input_files(args):
    if bool(args.input_path) == bool(args.input_dir):
        raise ValueError('Specify exactly one of --input_path or --input_dir')
    if args.input_path:
        paths = [Path(args.input_path)]
        common = paths[0].parent
    else:
        common = Path(args.input_dir)
        paths = sorted(common.rglob('*.tritsp') if args.recursive else common.glob('*.tritsp'))
    if not paths:
        raise ValueError('No *.tritsp files found. Use --recursive for nested scene folders.')
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths, common


def read_container(path):
    """Parse new, explicitly versioned format. Legacy *.diffc is rejected."""
    with open(path, 'rb') as f:
        prefix = f.read(HEADER.size)
        if len(prefix) != HEADER.size:
            raise ValueError(f'Truncated or legacy format (expected {HEADER.size}-byte header): {path}')
        (magic, version, mode_idx, preprocess_idx, flags, stop_idx, timestep,
         tsp_bytes, *rest) = HEADER.unpack(prefix)
        if magic != MAGIC or version != VERSION:
            raise ValueError(f'Unsupported file format: {path}. It must be produced by the new compress.py')
        if mode_idx >= len(MODES) or preprocess_idx >= len(PREPROCESS):
            raise ValueError('Invalid mode or preprocess selector')
        if flags & ~(RGB_FLAG | THERMAL_FLAG | TSP_FLAG):
            raise ValueError('Unknown flags in compressed file')
        if not flags & (RGB_FLAG | THERMAL_FLAG):
            raise ValueError('Compressed file contains no image stream')
        shape_vals = tuple(rest[:8])
        rgb_shape, thermal_shape = shape_vals[:4], shape_vals[4:]
        rgb_seed, thermal_seed, rgb_len, thermal_len = rest[8:12]
        cfg_sig, ckpt_sig, cache_sig = rest[12:15]
        mode = MODES[mode_idx]
        has_tsp = True
        if has_tsp != bool(flags & TSP_FLAG):
            raise ValueError('TSP presence does not match saved condition mode')
        if bool(flags & TSP_FLAG) != bool(tsp_bytes):
            raise ValueError('TSP flags / payload length mismatch')
        if bool(flags & RGB_FLAG) != bool(rgb_len) or bool(flags & THERMAL_FLAG) != bool(thermal_len):
            raise ValueError('Image stream flags / payload lengths mismatch')
        for present, shape in ((bool(flags & RGB_FLAG), rgb_shape),
                               (bool(flags & THERMAL_FLAG), thermal_shape)):
            if present and any(v == 0 for v in shape):
                raise ValueError('Invalid zero image dimensions')
            if not present and any(shape):
                raise ValueError('Absent stream has nonzero dimensions')
        if f.seek(0, 2) != HEADER.size + tsp_bytes + rgb_len + thermal_len:
            raise ValueError('Truncated or extra bytes in archive')
        f.seek(HEADER.size)
        tsp_stream = f.read(tsp_bytes)
        rgb_stream = f.read(rgb_len)
        thermal_stream = f.read(thermal_len)
    codes = unpack_tsp(tsp_stream) if tsp_bytes else None
    return {
        'mode': mode, 'preprocess_mode': PREPROCESS[preprocess_idx],
        'flags': flags, 'stop_idx': stop_idx, 'timestep': timestep,
        'rgb_shape': rgb_shape, 'thermal_shape': thermal_shape,
        'rgb_seed': rgb_seed, 'thermal_seed': thermal_seed,
        'rgb_stream': rgb_stream, 'thermal_stream': thermal_stream,
        'codes': codes, 'cfg_sig': cfg_sig, 'ckpt_sig': ckpt_sig, 'cache_sig': cache_sig,
        'file_bits': Path(path).stat().st_size * 8,
    }


def config_for_archive(path, args, container):
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = edict(yaml.safe_load(f))
    schedule = [int(x) for x in cfg.encoding_timesteps]
    stop_idx = container['stop_idx']
    if stop_idx >= len(schedule) or schedule[stop_idx] != container['timestep']:
        raise ValueError(f'DiffC stop timestep does not match config: {path}, '
                         f'stored index={stop_idx}, t={container["timestep"]}')
    manual = cfg.get('manual_dkl_per_step')
    if manual is None or len(manual) != len(schedule):
        raise ValueError('Decompress requires manual_dkl_per_step matching the full encoding schedule')
    cfg.encoding_timesteps = schedule[:stop_idx + 1]
    cfg.manual_dkl_per_step = list(manual[:stop_idx + 1])
    args.preprocess_mode = container['preprocess_mode']
    return cfg


def verify_container(container, cfg, args, mode, checkpoint_hash, cache_hash):
    if container['mode'] != mode:
        raise ValueError(f'File condition mode={container["mode"]} differs from model mode={mode}. '
                         'Use the matching trained checkpoint and mode.')
    if args.ignore_fingerprints:
        print('[UNSAFE] Ignoring model and config signatures; decoding may be incorrect')
        return
    expected = digest_config(cfg, args, mode)
    if container['cfg_sig'] != expected:
        raise ValueError('Codec settings mismatch! Ensure SAME encoding/denoising timesteps, DKL, '
                         'guidance scales, chunk params, prompt-stage boundaries and adapter args.')
    if container['ckpt_sig'] != checkpoint_hash:
        raise ValueError('Adapter checkpoint differs from the one used to encode this bitstream')
    if container['cache_sig'] != cache_hash:
        raise ValueError('Frozen text-cache differs from the one used to encode this bitstream')


def decode_modality(modality, container, config, model, simulator, args, output):
    stream = container[f'{modality}_stream']
    shape = container[f'{modality}_shape']
    seed = container[f'{modality}_seed']
    if not stream:
        return None
    raw_w, raw_h, proc_w, proc_h = shape
    
    gpu_sync(args.device)
    t0 = time.perf_counter()
    model.set_condition(container['codes'], modality, enabled=True,
                                 condition_mode=container['mode'])


    model.configure_empty(config.encoding_guidance_scale, proc_w, proc_h)
    with torch.inference_mode():
        seeds = simulator.decompress_chunk_seeds(
            list(stream), config.manual_dkl_per_step)
        latent = decode(
            proc_w, proc_h, config.encoding_timesteps, model,
            simulator, seeds, config.manual_dkl_per_step, seed=int(seed))
        model.configure_empty(config.denoising_guidance_scale, proc_w, proc_h)
        recon_latent = model.denoise(latent, container['timestep'], config.denoising_timesteps)
        if not torch.isfinite(recon_latent).all():
            raise FloatingPointError(f'{modality} reconstructed latent has NaN/Inf')
        recon = model.latent_to_image(recon_latent)
        if not torch.isfinite(recon).all():
            raise FloatingPointError(f'{modality} reconstructed image has NaN/Inf')
        if container['preprocess_mode'] == 'resize':
            
            
            restored = recon
        else:
            if proc_w < raw_w or proc_h < raw_h:
                raise ValueError(f'Processed dimensions {proc_w}x{proc_h} < raw image size {raw_w}x{raw_h}')
            restored = crop_to_raw(recon, (raw_h, raw_w))
        if modality == 'thermal':
            
            thermal_gray = gray3(restored)
            array = (thermal_gray[0].mean(dim=0).detach().float().cpu().numpy() * 255.0)
            array = array.round().clip(0, 255).astype('uint8')
            image = Image.fromarray(array, mode='L')
        else:
            image = image_utils.torch_to_pil_img(restored)
        gpu_sync(args.device)
        elapsed = time.perf_counter() - t0
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return elapsed



def main():
    args = parse_args()
    if args.text_cache is None:
        args.text_cache = str(Path(args.checkpoint).parent / 'text_cache.pt')
    rank, world_size = setup_parallel(args)
    if args.tritsp_tsp_condition_tokens + args.tritsp_spp_condition_tokens != args.tritsp_condition_tokens:
        raise ValueError('TSP queries + SPP queries must equal all condition queries')
    ckpt_path, text_cache_path = Path(args.checkpoint), Path(args.text_cache)
    for p in (ckpt_path, text_cache_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    mode = canonical_mode(checkpoint)
    del checkpoint
    files, root = input_files(args)
    local_files = [(i, path) for i, path in enumerate(files) if i % world_size == rank]
    ckpt_hash, cache_hash = digest_file(ckpt_path), digest_file(text_cache_path)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    params = None
    if local_files:
        first_path = local_files[0][1]
        container_first = read_container(first_path)
        cfg_first = config_for_archive(first_path, args, container_first)
        verify_container(container_first, cfg_first, args, mode, ckpt_hash, cache_hash)
        model = build_model(cfg_first, args)
        model.tritsp_text_cache = TriTSPTextCache.load(text_cache_path)
        model.prepare_base_prompt_cache([
            cfg_first.encoding_guidance_scale, cfg_first.denoising_guidance_scale
        ])
        model.offload_text_encoder()
        if rank == 0:
            params = benchmark_parameters_dup(model, mode)
        model.tritsp_adapter.eval()
        for i, path in local_files:
            gpu_sync(args.device)
            pair_decode_start = time.perf_counter()
            container = read_container(path)
            cfg = cfg_first if i == local_files[0][0] else config_for_archive(path, args, container)
            verify_container(container, cfg, args, mode, ckpt_hash, cache_hash)
            relative = path.relative_to(root)
            base = relative.with_suffix('')
            simulator = GaussianChannelSimulator(cfg.max_chunk_size, cfg.chunk_padding)
            rgb_path = out / 'rgb' / base.parent / f'{base.name}_decompressed.png'
            thermal_path = out / 'thermal' / base.parent / f'{base.name}_decompressed.png'
            r_time = decode_modality('rgb', container, cfg, model, simulator, args, rgb_path)
            t_time = decode_modality('thermal', container, cfg, model, simulator, args, thermal_path)
            gpu_sync(args.device)
            pair_decode_s = time.perf_counter() - pair_decode_start
            rows.append({
                '_file_index': i,
                'input_file': str(path), 'mode': mode, 'recon_timestep': container['timestep'],
                'rgb_output': str(rgb_path) if container['rgb_stream'] else '',
                'thermal_output': str(thermal_path) if container['thermal_stream'] else '',
                'rgb_decode_s': r_time, 'thermal_decode_s': t_time,
                'decode_pair_s': pair_decode_s,
                'total_file_bits': container['file_bits'],
            })
    rows = collect_rows(rows, world_size)
    if rank == 0:
        rows.sort(key=lambda row: row['_file_index'])
        for row in rows:
            del row['_file_index']
        with open(out / 'decompression_manifest.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'input_file', 'mode', 'recon_timestep', 'rgb_output', 'thermal_output',
                'rgb_decode_s', 'thermal_decode_s', 'decode_pair_s', 'total_file_bits'
            ])
            writer.writeheader()
            writer.writerows(rows)
        summary = write_benchmark_summary_dup(
            out / 'benchmark_decode_summary.json', rows, params,
            ['rgb_decode_s', 'thermal_decode_s', 'decode_pair_s']
        )
        print(f"[params] total={params['total_parameters']/1e6:.3f}M "
              f"trainable={params['trainable_parameters']/1e6:.3f}M")
        print(f"[decompress] mean RGB={summary['average_rgb_decode_s']}s "
              f"Thermal={summary['average_thermal_decode_s']}s")
        print(f'[decompress] Done: {len(rows)} archive(s); image outputs in {out}')
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
