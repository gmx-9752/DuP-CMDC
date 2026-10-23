# -*- coding: utf-8 -*-


from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.dup_cmdc.stage_prompt import STAGE_TO_INDEX_DUP_CMDC
from lib.dup_cmdc.tsp_core import FIELD_TYPE_IDS_DUP_CMDC


class QueryConditionDecoderDuPCmdc(nn.Module):
    """One shared condition decoder with two independent query banks/memories.

    The TransformerDecoderLayer parameters are shared between TSP and SPP.  The
    two priors never attend to a concatenated memory: each call receives only its
    own memory and its own learned query bank.
    """

    PRIOR_TSP = 0
    PRIOR_SPP = 1

    def __init__(
        self,
        text_dim: int,
        cond_dim: int = 512,
        num_queries: int = 12,
        num_tsp_queries: int = 8,
        num_spp_queries: int = 4,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.num_queries = int(num_queries)
        self.num_tsp_queries = int(num_tsp_queries)
        self.num_spp_queries = int(num_spp_queries)
        if self.num_tsp_queries <= 0 or self.num_spp_queries <= 0:
            raise ValueError("TSP/SPP query counts must both be positive")
        if self.num_tsp_queries + self.num_spp_queries != self.num_queries:
            raise ValueError(
                f"num_tsp_queries({self.num_tsp_queries}) + num_spp_queries({self.num_spp_queries}) "
                f"must equal num_queries({self.num_queries})"
            )

        
        self.text_proj = nn.Sequential(
            nn.LayerNorm(int(text_dim)),
            nn.Linear(int(text_dim), self.cond_dim),
        )
        self.position_embed = nn.Embedding(128, self.cond_dim)
        self.prior_embed = nn.Embedding(2, self.cond_dim)
        self.modality_embed = nn.Embedding(2, self.cond_dim)
        self.stage_embed = nn.Embedding(3, self.cond_dim)  
        self.tsp_field_embed = nn.Embedding(
            len(FIELD_TYPE_IDS_DUP_CMDC), self.cond_dim
        )

        
        self.tsp_queries = nn.Parameter(
            torch.randn(self.num_tsp_queries, self.cond_dim) * 0.02
        )
        self.spp_queries = nn.Parameter(
            torch.randn(self.num_spp_queries, self.cond_dim) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=self.cond_dim,
                    nhead=int(num_heads),
                    dim_feedforward=4 * self.cond_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.out_norm = nn.LayerNorm(self.cond_dim)

    def _modality_bias(
        self, batch_size: int, device: torch.device, modality_index: int
    ) -> torch.Tensor:
        mod_id = torch.full(
            (int(batch_size),), int(modality_index), device=device, dtype=torch.long
        )
        return self.modality_embed(mod_id).float()

    def _stage_bias(
        self, batch_size: int, device: torch.device, stage_index: int
    ) -> torch.Tensor:
        stage_id = torch.full(
            (int(batch_size),), int(stage_index), device=device, dtype=torch.long
        )
        return self.stage_embed(stage_id).float()

    def _add_positions(self, memory: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(memory.shape[1], device=memory.device, dtype=torch.long)
        pos = pos.clamp_max(self.position_embed.num_embeddings - 1)
        return memory + self.position_embed(pos).unsqueeze(0).to(dtype=memory.dtype)

    def build_tsp_memory(
        self,
        tsp_text: torch.Tensor,
        modality_index: int,
        tsp_field_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build image-specific TSP memory. No stage embedding is allowed here."""
        if tsp_text.ndim != 3:
            raise ValueError(f"tsp_text must be [B,T,D], got {tuple(tsp_text.shape)}")
        b, n_tsp = int(tsp_text.shape[0]), int(tsp_text.shape[1])
        if n_tsp <= 0:
            raise ValueError("TSP memory requested with zero TSP tokens")
        memory = self.text_proj(tsp_text.float())

        if tsp_field_ids is None:
            field_ids = torch.full(
                (b, n_tsp),
                FIELD_TYPE_IDS_DUP_CMDC["G"],
                device=memory.device,
                dtype=torch.long,
            )
        else:
            field_ids = tsp_field_ids.to(device=memory.device, dtype=torch.long)
            if field_ids.ndim == 1:
                if int(field_ids.numel()) != n_tsp:
                    raise ValueError(
                        f"Expected {n_tsp} TSP field ids, got {field_ids.numel()}"
                    )
                field_ids = field_ids.unsqueeze(0).expand(b, -1)
            elif field_ids.ndim == 2:
                if tuple(field_ids.shape) != (b, n_tsp):
                    if int(field_ids.shape[0]) == 1 and int(field_ids.shape[1]) == n_tsp:
                        field_ids = field_ids.expand(b, -1)
                    else:
                        raise ValueError(
                            f"Expected TSP field ids shape {(b, n_tsp)}, got {tuple(field_ids.shape)}"
                        )
            else:
                raise ValueError(
                    f"tsp_field_ids must be [T] or [B,T], got {tuple(field_ids.shape)}"
                )
        memory = memory + self.tsp_field_embed(field_ids).to(dtype=memory.dtype)
        memory = memory + self.prior_embed.weight[self.PRIOR_TSP].view(1, 1, -1).to(
            device=memory.device, dtype=memory.dtype
        )
        memory = self._add_positions(memory)
        mod_bias = self._modality_bias(b, memory.device, modality_index).to(memory.dtype)
        return memory + mod_bias.unsqueeze(1)

    def build_spp_memory(
        self,
        stage_text: torch.Tensor,
        stage_index: int,
        modality_index: int,
    ) -> torch.Tensor:
        """Build stage-specific SPP memory. This is the only branch with stage embedding."""
        if stage_text.ndim != 3:
            raise ValueError(
                f"stage_text must be [B,T,D], got {tuple(stage_text.shape)}"
            )
        b, n_spp = int(stage_text.shape[0]), int(stage_text.shape[1])
        if n_spp <= 0:
            raise ValueError("SPP memory requested with zero Stage Prompt tokens")
        memory = self.text_proj(stage_text.float())
        memory = memory + self.prior_embed.weight[self.PRIOR_SPP].view(1, 1, -1).to(
            device=memory.device, dtype=memory.dtype
        )
        memory = self._add_positions(memory)
        context = self._modality_bias(b, memory.device, modality_index)
        context = context + self._stage_bias(b, memory.device, stage_index)
        return memory + context.to(memory.dtype).unsqueeze(1)

    def _decode(
        self,
        memory: torch.Tensor,
        queries: torch.Tensor,
        context_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = int(memory.shape[0])
        q = queries.unsqueeze(0).expand(b, -1, -1).to(dtype=memory.dtype)
        q = q + context_bias.unsqueeze(1).to(dtype=memory.dtype)
        for layer in self.layers:
            q = layer(q, memory)
        q = self.out_norm(q)
        return q, q.mean(dim=1)

    def decode_tsp(
        self,
        tsp_text: torch.Tensor,
        modality_index: int,
        tsp_field_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        memory = self.build_tsp_memory(tsp_text, modality_index, tsp_field_ids)
        context = self._modality_bias(
            int(memory.shape[0]), memory.device, modality_index
        ).to(memory.dtype)
        return self._decode(memory, self.tsp_queries, context)

    def decode_spp(
        self,
        stage_text: torch.Tensor,
        stage_index: int,
        modality_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        memory = self.build_spp_memory(stage_text, stage_index, modality_index)
        context = self._modality_bias(
            int(memory.shape[0]), memory.device, modality_index
        ) + self._stage_bias(int(memory.shape[0]), memory.device, stage_index)
        return self._decode(memory, self.spp_queries, context.to(memory.dtype))


class PromptCrossAttentionDuPCmdc(nn.Module):
    """TSP-only visual Cross-Attention branch."""

    def __init__(self, hidden_dim: int, cond_dim: int, num_heads: int = 8):
        super().__init__()
        if int(cond_dim) % int(num_heads) != 0:
            raise ValueError("cond_dim must be divisible by num_heads")
        self.hidden_norm = nn.LayerNorm(int(hidden_dim))
        self.cond_norm = nn.LayerNorm(int(cond_dim))
        self.q_proj = nn.Linear(int(hidden_dim), int(cond_dim), bias=False)
        self.attn = nn.MultiheadAttention(
            int(cond_dim), int(num_heads), dropout=0.0, batch_first=True
        )
        self.out_proj = nn.Linear(int(cond_dim), int(hidden_dim), bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(self.hidden_norm(hidden.float()))
        c = self.cond_norm(condition.float())
        out, _ = self.attn(q, c, c, need_weights=False)
        return self.out_proj(out)


class FiLMDuPCmdc(nn.Module):
    """SPP-only stage modulation branch."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.mlp = nn.Sequential(
            nn.LayerNorm(int(cond_dim)),
            nn.Linear(int(cond_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 2 * int(hidden_dim)),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, hidden: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mlp(summary.float()).chunk(2, dim=-1)
        return (
            self.norm(hidden.float()) * torch.tanh(scale).unsqueeze(1)
            + shift.unsqueeze(1)
        )


class LowRankDeltaDuPCmdc(nn.Module):
    def __init__(self, hidden_dim: int, rank: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.down = nn.Linear(int(hidden_dim), int(rank), bias=False)
        self.up = nn.Linear(int(rank), int(hidden_dim), bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(self.norm(hidden.float()))))


class DuPCmdcBridgeAdapter(nn.Module):
    """Split-prior DPMA with three fully decoupled residual paths."""

    ARCHITECTURE_VERSION = "dup_cmdc_full_v1"

    def __init__(
        self,
        hidden_dim: int,
        text_dim: int,
        cond_dim: int = 512,
        num_injection_layers: int = 4,
        num_heads: int = 8,
        num_condition_tokens: int = 12,
        num_tsp_condition_tokens: int = 8,
        num_spp_condition_tokens: int = 4,
        condition_decoder_layers: int = 2,
        modality_delta_rank: int = 64,
        gate_max: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        self.num_injection_layers = int(num_injection_layers)
        self.gate_max = float(gate_max)
        self.num_condition_tokens = int(num_condition_tokens)
        self.num_tsp_condition_tokens = int(num_tsp_condition_tokens)
        self.num_spp_condition_tokens = int(num_spp_condition_tokens)

        self.condition_decoder = QueryConditionDecoderDuPCmdc(
            text_dim=int(text_dim),
            cond_dim=self.cond_dim,
            num_queries=self.num_condition_tokens,
            num_tsp_queries=self.num_tsp_condition_tokens,
            num_spp_queries=self.num_spp_condition_tokens,
            num_heads=int(num_heads),
            num_layers=int(condition_decoder_layers),
        )
        self.cross_layers = nn.ModuleList(
            [
                PromptCrossAttentionDuPCmdc(
                    self.hidden_dim, self.cond_dim, num_heads=int(num_heads)
                )
                for _ in range(self.num_injection_layers)
            ]
        )
        self.film_layers = nn.ModuleList(
            [
                FiLMDuPCmdc(self.hidden_dim, self.cond_dim)
                for _ in range(self.num_injection_layers)
            ]
        )
        self.rgb_deltas = nn.ModuleList(
            [
                LowRankDeltaDuPCmdc(
                    self.hidden_dim, rank=int(modality_delta_rank)
                )
                for _ in range(self.num_injection_layers)
            ]
        )
        self.thermal_deltas = nn.ModuleList(
            [
                LowRankDeltaDuPCmdc(
                    self.hidden_dim, rank=int(modality_delta_rank)
                )
                for _ in range(self.num_injection_layers)
            ]
        )


        self.tsp_gate_logits = nn.Parameter(
            torch.full((self.num_injection_layers, 2), -4.0)
        )
        
        self.spp_gate_logits = nn.Parameter(
            torch.full((self.num_injection_layers, 3, 2), -4.0)
        )
        
        self.delta_gate_logits = nn.Parameter(
            torch.full((self.num_injection_layers, 2), -4.0)
        )

        self._last_aux: Dict[str, object] = {}
        self._branch_logging_enabled = False
        self._branch_stats: Dict[Tuple[int, str, str, str], Dict[str, float]] = {}
        self._structural_mode = "full"

    @staticmethod
    def modality_index(modality: str) -> int:
        return 1 if str(modality).lower() in {"thermal", "ir", "t"} else 0

    @staticmethod
    def _normalize_condition_mode(condition_mode: str) -> str:
        if str(condition_mode).strip().lower() != "full":
            raise ValueError("Only full mode is supported")
        return "full"

    @classmethod
    def _mode_spec(cls, condition_mode: str) -> Dict[str, object]:
        cls._normalize_condition_mode(condition_mode)
        return {"mode": "full", "use_tsp_text": True, "use_stage_text": True,
                "tsp_cross": True, "spp_film": True, "delta": True}


    @classmethod
    def _hard_mask_prompt_inputs(
        cls, stage_text: torch.Tensor, tsp_text: torch.Tensor,
        tsp_field_ids: Optional[torch.Tensor], condition_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str]:
        cls._normalize_condition_mode(condition_mode)
        if stage_text.ndim != 3 or tsp_text.ndim != 3:
            raise ValueError("Stage and TSP prompts must be rank-3 tensors")
        if stage_text.shape[0] != tsp_text.shape[0]:
            raise ValueError("Stage and TSP batch sizes differ")
        return stage_text, tsp_text, tsp_field_ids, "full"

    def _gate_tsp(self, slot: int, modality: str) -> torch.Tensor:
        m = self.modality_index(modality)
        return self.gate_max * torch.sigmoid(self.tsp_gate_logits[int(slot), m])

    def _gate_spp(self, slot: int, stage: str, modality: str) -> torch.Tensor:
        m = self.modality_index(modality)
        s = STAGE_TO_INDEX_DUP_CMDC[str(stage)]
        return self.gate_max * torch.sigmoid(self.spp_gate_logits[int(slot), s, m])

    def _gate_delta(self, slot: int, modality: str) -> torch.Tensor:
        m = self.modality_index(modality)
        return self.gate_max * torch.sigmoid(self.delta_gate_logits[int(slot), m])

    def configure_trainable_for_mode(self, condition_mode: str = "full") -> Dict[str, int]:
        self._normalize_condition_mode(condition_mode)
        self._structural_mode = "full"
        for parameter in self.parameters():
            parameter.requires_grad = True
        return {"trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "total": sum(p.numel() for p in self.parameters())}

    def set_lowrank_trainable(self, enabled: bool, condition_mode: Optional[str] = None) -> None:
        self._normalize_condition_mode("full" if condition_mode is None else condition_mode)
        for module in list(self.rgb_deltas) + list(self.thermal_deltas):
            for parameter in module.parameters():
                parameter.requires_grad = bool(enabled)
        self.delta_gate_logits.requires_grad = bool(enabled)


    
    def set_branch_logging(self, enabled: bool) -> None:
        self._branch_logging_enabled = bool(enabled)

    def reset_branch_contribution_stats(self) -> None:
        self._branch_stats = {}

    @torch.no_grad()
    def _record_branch_contribution(
        self,
        *,
        hidden: torch.Tensor,
        tsp_residual: Optional[torch.Tensor],
        spp_residual: Optional[torch.Tensor],
        delta_residual: Optional[torch.Tensor],
        slot: int,
        stage: str,
        modality: str,
        mode: str,
        gate_tsp: float,
        gate_spp: float,
        gate_delta: float,
    ) -> None:
        if not self._branch_logging_enabled:
            return
        h = hidden.detach().float()
        denom = torch.linalg.vector_norm(h).clamp_min(1e-12)

        def ratio(x: Optional[torch.Tensor]) -> float:
            if x is None:
                return 0.0
            return float(
                (torch.linalg.vector_norm(x.detach().float()) / denom).cpu().item()
            )

        key = (int(slot), str(stage), str(modality), str(mode))
        if key not in self._branch_stats:
            self._branch_stats[key] = defaultdict(float)  
        stats = self._branch_stats[key]
        stats["count"] += 1.0
        stats["r_tsp_sum"] += ratio(tsp_residual)
        stats["r_spp_sum"] += ratio(spp_residual)
        stats["r_delta_sum"] += ratio(delta_residual)
        stats["gate_tsp_sum"] += float(gate_tsp)
        stats["gate_spp_sum"] += float(gate_spp)
        stats["gate_delta_sum"] += float(gate_delta)

    def consume_branch_contribution_stats(self, reset: bool = True) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for (slot, stage, modality, mode), stats in sorted(self._branch_stats.items()):
            count = max(float(stats.get("count", 0.0)), 1.0)
            rows.append(
                {
                    "slot": int(slot),
                    "stage": str(stage),
                    "modality": str(modality),
                    "condition_mode": str(mode),
                    "count": int(stats.get("count", 0.0)),
                    "r_tsp": float(stats.get("r_tsp_sum", 0.0)) / count,
                    "r_spp": float(stats.get("r_spp_sum", 0.0)) / count,
                    "r_delta": float(stats.get("r_delta_sum", 0.0)) / count,
                    "gate_tsp": float(stats.get("gate_tsp_sum", 0.0)) / count,
                    "gate_spp": float(stats.get("gate_spp_sum", 0.0)) / count,
                    "gate_delta": float(stats.get("gate_delta_sum", 0.0)) / count,
                }
            )
        if reset:
            self.reset_branch_contribution_stats()
        return rows

    def apply_to_tokens(
        self, hidden: torch.Tensor, slot: int, stage: str, modality: str,
        stage_text: torch.Tensor, tsp_text: torch.Tensor,
        tsp_field_ids: Optional[torch.Tensor] = None,
        condition_mode: str = "full",
    ) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim:
            raise RuntimeError(f"DPMA expected [B,N,{self.hidden_dim}], got {tuple(hidden.shape)}")
        stage_text, tsp_text, tsp_field_ids, mode = self._hard_mask_prompt_inputs(
            stage_text, tsp_text, tsp_field_ids, condition_mode
        )
        stage_idx = STAGE_TO_INDEX_DUP_CMDC[str(stage)]
        mod_idx = self.modality_index(modality)
        g_t = self._gate_tsp(slot, modality).to(hidden.device, torch.float32)
        g_s = self._gate_spp(slot, stage, modality).to(hidden.device, torch.float32)
        g_d = self._gate_delta(slot, modality).to(hidden.device, torch.float32)
        tsp_condition, _ = self.condition_decoder.decode_tsp(
            tsp_text=tsp_text, modality_index=mod_idx, tsp_field_ids=tsp_field_ids
        )
        tsp_attn = self.cross_layers[int(slot)](hidden, tsp_condition)
        _, spp_summary = self.condition_decoder.decode_spp(
            stage_text=stage_text, stage_index=stage_idx, modality_index=mod_idx
        )
        spp_film = self.film_layers[int(slot)](hidden, spp_summary)
        delta = (self.thermal_deltas[int(slot)](hidden) if mod_idx == 1
                 else self.rgb_deltas[int(slot)](hidden))
        tsp_residual = g_t * tsp_attn
        spp_residual = g_s * spp_film
        delta_residual = g_d * delta
        out = hidden.float() + tsp_residual + spp_residual + delta_residual
        self._record_branch_contribution(
            hidden=hidden, tsp_residual=tsp_residual, spp_residual=spp_residual,
            delta_residual=delta_residual, slot=slot, stage=stage, modality=modality,
            mode=mode, gate_tsp=float(g_t.detach().cpu()),
            gate_spp=float(g_s.detach().cpu()), gate_delta=float(g_d.detach().cpu()),
        )
        self._last_aux = {
            "gate_tsp": float(g_t.detach().cpu()),
            "gate_spp": float(g_s.detach().cpu()),
            "gate_delta": float(g_d.detach().cpu()),
            "gate_attn": float(g_t.detach().cpu()),
            "gate_film": float(g_s.detach().cpu()),
            "stage_index": float(stage_idx), "modality_index": float(mod_idx),
            "condition_mode": "full", "stage_tokens_used": float(stage_text.shape[1]),
            "tsp_tokens_used": float(tsp_text.shape[1]),
        }
        return out.to(dtype=hidden.dtype)

    def gate_regularization(self, condition_mode: str = "full") -> torch.Tensor:
        self._normalize_condition_mode(condition_mode)
        return torch.stack([
            torch.sigmoid(self.tsp_gate_logits.float()).pow(2).mean(),
            torch.sigmoid(self.spp_gate_logits.float()).pow(2).mean(),
            torch.sigmoid(self.delta_gate_logits.float()).pow(2).mean(),
        ]).mean()

    def delta_regularization(self, condition_mode: str = "full") -> torch.Tensor:
        self._normalize_condition_mode(condition_mode)
        return torch.stack([
            module.up.weight.float().pow(2).mean()
            for module in list(self.rgb_deltas) + list(self.thermal_deltas)
        ]).mean()

    def get_last_aux(self) -> Dict[str, object]:
        return dict(self._last_aux)


DuPCmdcBridgeAdapter = DuPCmdcBridgeAdapter
