# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from datetime import timedelta
from typing import Dict, Iterable

import torch
import torch.distributed as dist


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(hours=3))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if distributed else 0)
    return distributed, rank, local_rank, world_size


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_parameters(parameters: Iterable[torch.nn.Parameter], src: int = 0) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in parameters:
        dist.broadcast(p.data, src=int(src))


def average_gradients(parameters: Iterable[torch.nn.Parameter], world_size: int) -> None:
    if int(world_size) <= 1:
        return
    for p in parameters:


        if not p.requires_grad:
            continue


        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(float(world_size))


def reduce_scalar_dict(values: Dict[str, float], count: float = 1.0) -> Dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return dict(values)
    keys = sorted(values)
    device = torch.device("cuda", torch.cuda.current_device())
    tensor = torch.tensor([float(values[k]) for k in keys] + [float(count)], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_count = max(float(tensor[-1].item()), 1e-12)
    return {k: float(tensor[i].item()) / total_count for i, k in enumerate(keys)}


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
