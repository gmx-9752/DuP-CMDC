# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

from lib.dup_cmdc.stage_prompt import stage_prompt_text
from lib.dup_cmdc.tsp_core import (
    FIELD_TYPE_IDS_DUP_CMDC,
    iter_tsp_primitive_specs,
    normalize_tsp_core,
    primitive_text,
)
from lib.dup_cmdc.tsp_schema import (
    ENVIRONMENT_CODES,
    GEOMETRY_CODES,
    MORPHOLOGY_CODES,
    ORGANIZATION_CODES,
    REGION_CODES,
    TOPOLOGY_CODES,
)


def segment_pool_text(embeds: torch.Tensor, mask: Optional[torch.Tensor], num_tokens: int) -> torch.Tensor:
    if embeds.ndim != 3 or embeds.shape[0] != 1:
        raise ValueError(f"Expected prompt embeds [1,L,D], got {tuple(embeds.shape)}")
    x = embeds[0].float()
    valid = x if mask is None else x[mask[0].to(dtype=torch.bool, device=x.device)]
    if valid.shape[0] == 0:
        valid = x[:1]
    chunks = torch.tensor_split(valid, int(num_tokens), dim=0)
    pooled = [chunk.mean(dim=0) if chunk.shape[0] else valid.mean(dim=0) for chunk in chunks]
    while len(pooled) < int(num_tokens):
        pooled.append(pooled[-1].clone())
    return torch.stack(pooled[: int(num_tokens)], dim=0).contiguous()


class DuPCmdcTextCache:
    """Versioned, CPU-resident cache for stage and structured primitive tokens."""

    CACHE_VERSION = "TriTSPXiaorong structured primitive text cache v1"

    def __init__(self, num_stage_tokens: int = 8, num_primitive_tokens: int = 2, num_field_tokens: Optional[int] = None):
        self.num_stage_tokens = int(num_stage_tokens)
        # ``num_field_tokens`` is accepted as a compatibility spelling, but it
        # refers to each primitive rather than to an old fixed-field layout.
        self.num_primitive_tokens = int(num_field_tokens if num_field_tokens is not None else num_primitive_tokens)
        self.num_field_tokens = self.num_primitive_tokens
        self.text_dim: Optional[int] = None
        self.entries: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _modality(modality: str) -> str:
        return "thermal" if str(modality).lower() in {"ir", "thermal", "t"} else "rgb"

    @classmethod
    def stage_key(cls, stage: str, modality: str) -> str:
        return f"stage::{str(stage).lower()}::{cls._modality(modality)}"

    @classmethod
    def primitive_key(cls, token_key: str, modality: str) -> str:
        return f"primitive::{str(token_key)}::{cls._modality(modality)}"

    def _encode_text(self, model, text: str, num_tokens: int) -> torch.Tensor:
        with torch.inference_mode():
            encoded = model.pipe.encode_prompt(
                prompt=str(text),
                do_classifier_free_guidance=False,
                negative_prompt="",
                device=torch.device(model.device),
                clean_caption=False,
                max_sequence_length=int(model.max_sequence_length),
                complex_human_instruction=False,
            )
        pooled = segment_pool_text(encoded[0], encoded[1], int(num_tokens))
        if self.text_dim is None:
            self.text_dim = int(pooled.shape[-1])
        elif int(pooled.shape[-1]) != int(self.text_dim):
            raise RuntimeError(f"Text dimension changed: {pooled.shape[-1]} vs {self.text_dim}")
        return pooled.cpu().to(dtype=torch.float16)

    def ensure_stage(self, model, stage: str, modality: str) -> None:
        key = self.stage_key(stage, modality)
        if key not in self.entries:
            self.entries[key] = self._encode_text(model, stage_prompt_text(stage, modality), self.num_stage_tokens)

    def _ensure_primitive(self, model, token_key: str, code: str, field_type: str, object_key: str, modality: str) -> None:
        key = self.primitive_key(token_key, modality)
        if key in self.entries:
            return
        phrase = primitive_text(token_key, code, field_type, object_key, modality)
        self.entries[key] = self._encode_text(model, phrase, self.num_primitive_tokens)

    def ensure_tsp(self, model, codes: Mapping[str, Any], modality: str) -> None:
        c = normalize_tsp_core(codes)
        for token_key, code, field_type, object_key in iter_tsp_primitive_specs(c):
            self._ensure_primitive(model, token_key, code, field_type, object_key, modality)

    def ensure_all(self, model, codes_iter: Iterable[Mapping[str, Any]]) -> None:
        for stage in ("high", "middle", "low"):
            for modality in ("rgb", "thermal"):
                self.ensure_stage(model, stage, modality)


        for modality in ("rgb", "thermal"):
            for code in GEOMETRY_CODES:
                self._ensure_primitive(model, code, code, "G", "", modality)
            for code in ENVIRONMENT_CODES:
                self._ensure_primitive(model, code, code, "E", "", modality)
            for token_key, family in (("C_H", "human"), ("C_V", "vehicle")):
                self._ensure_primitive(model, token_key, token_key, "C", family, modality)
            for prefix, object_key in (("H", "human"), ("V", "vehicle")):
                for code in ORGANIZATION_CODES:
                    self._ensure_primitive(model, f"{prefix}_{code}", code, "O", object_key, modality)
                for code in REGION_CODES:
                    self._ensure_primitive(model, f"{prefix}_{code}", code, "R", object_key, modality)
                for code in TOPOLOGY_CODES:
                    self._ensure_primitive(model, f"{prefix}_{code}", code, "T", object_key, modality)
                for code in MORPHOLOGY_CODES:
                    # M0/M1/... are already complete morphology codes.
                    self._ensure_primitive(model, f"{prefix}_{code}", code, "M", object_key, modality)

        for codes in codes_iter:
            normalized = normalize_tsp_core(codes)
            for modality in ("rgb", "thermal"):
                self.ensure_tsp(model, normalized, modality)

    def get_stage(self, stage: str, modality: str, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        key = self.stage_key(stage, modality)
        if key not in self.entries:
            raise KeyError(f"Stage text cache entry not found: {key}")
        return self.entries[key].to(device=device, dtype=dtype, non_blocking=True).unsqueeze(0)

    def get_tsp_with_field_ids(
        self,
        codes: Optional[Mapping[str, Any]],
        modality: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if codes is None:
            if self.text_dim is None:
                raise RuntimeError("Cannot create an empty TSP bank before text_dim is known")
            empty = torch.empty((1, 0, int(self.text_dim)), device=device, dtype=dtype)
            return empty, torch.empty((1, 0), device=device, dtype=torch.long)
        values = []
        field_ids = []
        for token_key, code, field_type, object_key in iter_tsp_primitive_specs(codes):
            key = self.primitive_key(token_key, modality)
            if key not in self.entries:
                raise KeyError(f"TSP primitive text cache entry not found: {key}")
            values.append(self.entries[key])
            field_ids.extend([FIELD_TYPE_IDS_DUP_CMDC[field_type]] * self.num_primitive_tokens)
        if not values:
            return self.get_tsp_with_field_ids(None, modality, device, dtype)
        tensor = torch.cat(values, dim=0).to(device=device, dtype=dtype, non_blocking=True).unsqueeze(0)
        ids = torch.tensor(field_ids, device=device, dtype=torch.long).unsqueeze(0)
        return tensor, ids

    def get_tsp(self, codes: Mapping[str, Any], modality: str, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.get_tsp_with_field_ids(codes, modality, device, dtype)[0]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": self.CACHE_VERSION,
            "num_stage_tokens": self.num_stage_tokens,
            "num_primitive_tokens": self.num_primitive_tokens,
            "text_dim": self.text_dim,
            "entries": self.entries,
        }, target)

    @classmethod
    def load(cls, path: str | Path) -> "DuPCmdcTextCache":
        obj = torch.load(Path(path), map_location="cpu")
        if obj.get("version") != cls.CACHE_VERSION:
            raise ValueError(f"Outdated or incompatible TriTSPXiaorong text cache: {path}; rebuild it")
        cache = cls(int(obj["num_stage_tokens"]), int(obj["num_primitive_tokens"]))
        cache.text_dim = int(obj["text_dim"])
        cache.entries = {str(k): v.detach().cpu() for k, v in obj["entries"].items()}
        return cache


TriTSPTextCache = DuPCmdcTextCache
