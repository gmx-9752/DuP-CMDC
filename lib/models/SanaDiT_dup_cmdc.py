# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn

from lib.models.SanaDiT import SanaDiTModel, _check_finite, ddpm_to_flow_factor
from lib.diffc.utils.alpha_beta import get_alpha_prod_and_beta_prod
from lib.dup_cmdc.adapter import DuPCmdcBridgeAdapter
from lib.dup_cmdc.stage_prompt import stage_from_timestep
from lib.dup_cmdc.text_cache import TriTSPTextCache
from lib.dup_cmdc.tsp_core import normalize_tsp_core


def _derive_hidden_dim(transformer: nn.Module, explicit: Optional[int] = None) -> int:
    if explicit is not None:
        return int(explicit)
    cfg = getattr(transformer, "config", None)
    for name in ("inner_dim", "hidden_size", "embed_dim", "caption_channels"):
        value = getattr(cfg, name, None) if cfg is not None else None
        if isinstance(value, int) and value > 0 and name != "caption_channels":
            return int(value)
    heads = getattr(cfg, "num_attention_heads", None) if cfg is not None else None
    head_dim = getattr(cfg, "attention_head_dim", None) if cfg is not None else None
    if isinstance(heads, int) and isinstance(head_dim, int):
        return int(heads) * int(head_dim)
    
    for module in transformer.modules():
        if isinstance(module, nn.Linear) and module.in_features >= 256 and module.in_features == module.out_features:
            return int(module.in_features)
    raise RuntimeError("Could not infer SANA hidden dimension; pass --tritsp_hidden_size explicitly")


def _find_blocks(transformer: nn.Module) -> List[nn.Module]:
    for name in ("transformer_blocks", "blocks", "layers"):
        value = getattr(transformer, name, None)
        if isinstance(value, (nn.ModuleList, list, tuple)) and len(value) > 0:
            return list(value)
    candidates = []
    for _, module in transformer.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 4:
            candidates.append(list(module))
    if not candidates:
        raise RuntimeError("Could not locate SANA transformer blocks")
    return max(candidates, key=len)


class SanaDiTModel_DuPCmdc(SanaDiTModel):
    def __init__(
        self,
        *args,
        tritsp_ckpt: Optional[str] = None,
        tritsp_text_cache_path: Optional[str] = None,
        tritsp_hidden_size: Optional[int] = None,
        tritsp_cond_dim: int = 512,
        tritsp_num_heads: int = 8,
        tritsp_condition_tokens: int = 12,
        tritsp_tsp_condition_tokens: int = 8,
        tritsp_spp_condition_tokens: int = 4,
        tritsp_condition_decoder_layers: int = 2,
        tritsp_delta_rank: int = 64,
        tritsp_gate_max: float = 0.5,
        tritsp_inject_block_indices: str = "9,14,18,23",
        tritsp_high_boundary: int = 300,
        tritsp_middle_boundary: int = 85,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.freeze_base_model()
        self.tritsp_enabled = True
        self.tritsp_codes: Optional[Dict[str, Any]] = None
        self.tritsp_modality = "rgb"
        self.tritsp_current_timestep = 0
        self.tritsp_use_stage_prompt = True
        self.tritsp_use_tsp_prompt = True
        self.tritsp_condition_mode = "full"
        self.tritsp_high_boundary = int(tritsp_high_boundary)
        self.tritsp_middle_boundary = int(tritsp_middle_boundary)
        self._hooks: List[Any] = []
        self._base_prompt_cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        if tritsp_text_cache_path and Path(tritsp_text_cache_path).exists():
            try:
                self.tritsp_text_cache = TriTSPTextCache.load(tritsp_text_cache_path)
            except ValueError:
                self.tritsp_text_cache = TriTSPTextCache(num_stage_tokens=8, num_field_tokens=2)
                self.tritsp_text_cache.ensure_stage(self, "high", "rgb")
        else:
            self.tritsp_text_cache = TriTSPTextCache(num_stage_tokens=8, num_field_tokens=2)
            self.tritsp_text_cache.ensure_stage(self, "high", "rgb")
        if self.tritsp_text_cache.text_dim is None:
            raise RuntimeError("TriTSP text cache has no text_dim")

        hidden_dim = _derive_hidden_dim(self.transformer, explicit=tritsp_hidden_size)
        blocks = _find_blocks(self.transformer)
        indices = [int(x) for x in str(tritsp_inject_block_indices).replace(",", " ").split()]
        indices = [i for i in indices if 0 <= i < len(blocks)]
        if not indices:
            raise ValueError(f"No valid TriTSP injection indices in {tritsp_inject_block_indices!r}; num_blocks={len(blocks)}")
        self.tritsp_inject_block_indices = indices
        self.tritsp_adapter = DuPCmdcBridgeAdapter(
            hidden_dim=hidden_dim,
            text_dim=int(self.tritsp_text_cache.text_dim),
            cond_dim=int(tritsp_cond_dim),
            num_injection_layers=len(indices),
            num_heads=int(tritsp_num_heads),
            num_condition_tokens=int(tritsp_condition_tokens),
            num_tsp_condition_tokens=int(tritsp_tsp_condition_tokens),
            num_spp_condition_tokens=int(tritsp_spp_condition_tokens),
            condition_decoder_layers=int(tritsp_condition_decoder_layers),
            modality_delta_rank=int(tritsp_delta_rank),
            gate_max=float(tritsp_gate_max),
        ).to(device=self.device, dtype=torch.float32)
        self._register_hooks(blocks, indices)
        if tritsp_ckpt:
            self.load_checkpoint(tritsp_ckpt)
        self.tritsp_adapter.train()
        print(
            f"[DuPCmdc] hidden_dim={hidden_dim}, text_dim={self.tritsp_text_cache.text_dim}, "
            f"inject_blocks={indices}, queries=TSP:{int(tritsp_tsp_condition_tokens)}+SPP:{int(tritsp_spp_condition_tokens)}, "
            f"device={self.device}"
        )

    def freeze_base_model(self) -> None:
        for module in (self.vae, self.transformer, self.pipe.text_encoder):
            for p in module.parameters():
                p.requires_grad = False

    def prebuild_text_cache(self, codes_iter: Iterable[Mapping[str, Any]], save_path: Optional[str | Path] = None) -> None:
        self.tritsp_text_cache.ensure_all(self, codes_iter)
        if save_path is not None:
            self.tritsp_text_cache.save(save_path)

    def prepare_base_prompt_cache(self, guidance_scales: Iterable[float]) -> None:
        for scale in sorted(set(float(x) for x in guidance_scales)):
            key = f"{scale:.8g}"
            if key in self._base_prompt_cache:
                continue
            super().configure("", scale, 64, 64)
            self._base_prompt_cache[key] = (
                self.prompt_embeds.detach().to("cpu", dtype=torch.float16),
                self.prompt_attention_mask.detach().to("cpu"),
            )

    def configure_empty(self, prompt_guidance: float, image_width: int, image_height: int) -> None:
        scale = float(prompt_guidance)
        key = f"{scale:.8g}"
        if key not in self._base_prompt_cache:
            self.prepare_base_prompt_cache([scale])
        embeds, mask = self._base_prompt_cache[key]
        self.guidance_scale = scale
        self.use_classifier_free_guidance = scale > 1.0
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.prompt_embeds = embeds.to(self.device, dtype=self.dtype, non_blocking=True)
        self.prompt_attention_mask = mask.to(self.device, non_blocking=True)

    def offload_text_encoder(self) -> None:
        try:
            self.pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            print("[DuPCmdc] Frozen text encoder moved to CPU after text-cache construction.")
        except Exception as exc:
            print(f"[DuPCmdc][WARN] Could not offload text encoder: {exc}")

    def set_enabled(self, enabled: bool) -> None:
        self.tritsp_enabled = bool(enabled)

    @staticmethod
    def normalize_condition_mode(mode: str) -> str:
        if str(mode).strip().lower() != "full":
            raise ValueError("Only full mode is supported")
        return "full"

    def set_condition(
        self, codes: Optional[Mapping[str, Any]], modality: str,
        enabled: Optional[bool] = None,
        condition_mode: Optional[str] = None,
    ) -> None:
        if condition_mode is not None:
            self.normalize_condition_mode(condition_mode)
        self.tritsp_modality = "thermal" if str(modality).lower() in {"thermal", "ir", "t"} else "rgb"
        if enabled is not None:
            self.tritsp_enabled = bool(enabled)
        self.tritsp_condition_mode = "full"
        self.tritsp_codes = None if codes is None else normalize_tsp_core(codes)

    def _register_hooks(self, blocks: List[nn.Module], indices: List[int]) -> None:
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._hooks.clear()
        for slot, block_idx in enumerate(indices):
            self._hooks.append(blocks[int(block_idx)].register_forward_hook(self._make_hook(slot)))

    def _make_hook(self, slot: int):
        def hook(_module, _inputs, output):
            if not self.tritsp_enabled:
                return output
            return self._apply_adapter_output(output, int(slot))
        return hook

    def _apply_adapter_tensor(self, x: torch.Tensor, slot: int) -> torch.Tensor:
        stage = stage_from_timestep(
            self.tritsp_current_timestep, self.tritsp_high_boundary, self.tritsp_middle_boundary
        )
        text_dim = int(self.tritsp_text_cache.text_dim)
        stage_text = self.tritsp_text_cache.get_stage(stage, self.tritsp_modality, x.device, torch.float32)
        tsp_text, tsp_field_ids = self.tritsp_text_cache.get_tsp_with_field_ids(
            self.tritsp_codes, self.tritsp_modality, x.device, torch.float32
        )
        batch_size = int(x.shape[0])
        if stage_text.shape[0] != batch_size:
            stage_text = stage_text.expand(batch_size, -1, -1)
            tsp_text = tsp_text.expand(batch_size, -1, -1)
            tsp_field_ids = tsp_field_ids.expand(batch_size, -1)
        if x.dim() == 3:
            return self.tritsp_adapter.apply_to_tokens(
                x, slot, stage, self.tritsp_modality, stage_text, tsp_text, tsp_field_ids,
                condition_mode=self.tritsp_condition_mode,
            )
        if x.dim() == 4:
            b = x.shape[0]
            if x.shape[1] == self.tritsp_adapter.hidden_dim:  
                c, h, w = x.shape[1:]
                tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
                y = self.tritsp_adapter.apply_to_tokens(
                    tokens, slot, stage, self.tritsp_modality, stage_text, tsp_text, tsp_field_ids,
                    condition_mode=self.tritsp_condition_mode,
                )
                return y.reshape(b, h, w, c).permute(0, 3, 1, 2).to(dtype=x.dtype)
            if x.shape[-1] == self.tritsp_adapter.hidden_dim:  
                h, w, c = x.shape[1:]
                tokens = x.reshape(b, h * w, c)
                y = self.tritsp_adapter.apply_to_tokens(
                    tokens, slot, stage, self.tritsp_modality, stage_text, tsp_text, tsp_field_ids,
                    condition_mode=self.tritsp_condition_mode,
                )
                return y.reshape(b, h, w, c).to(dtype=x.dtype)
        return x

    def _apply_adapter_output(self, output, slot: int):
        if torch.is_tensor(output):
            return self._apply_adapter_tensor(output, slot)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return (self._apply_adapter_tensor(output[0], slot),) + tuple(output[1:])
        if isinstance(output, list) and output and torch.is_tensor(output[0]):
            values = list(output)
            values[0] = self._apply_adapter_tensor(values[0], slot)
            return values
        return output

    def _predict_flow_impl(self, flow_latent: torch.Tensor, timestep: int) -> torch.Tensor:
        if self.prompt_embeds is None:
            raise RuntimeError("Call configure_empty before predict_noise/denoise")
        self.tritsp_current_timestep = int(timestep)
        flow_latent = flow_latent.to(device=self.device, dtype=self.dtype).contiguous()
        latent_model_input = torch.cat([flow_latent] * 2, dim=0) if self.use_classifier_free_guidance else flow_latent
        prompt_embeds = self.prompt_embeds.to(device=self.device, dtype=self.dtype).contiguous()
        prompt_mask = self.prompt_attention_mask.to(device=self.device)
        t = torch.full((latent_model_input.shape[0],), float(timestep), device=self.device, dtype=torch.float32)
        t = t * float(getattr(self.transformer.config, "timestep_scale", 1.0))
        cudnn_ctx = torch.backends.cudnn.flags(enabled=False) if self.disable_cudnn_for_transformer else contextlib.nullcontext()
        with cudnn_ctx:
            flow_pred = self.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_mask,
                timestep=t,
                return_dict=False,
            )[0].float()
        if flow_pred.shape[1] != flow_latent.shape[1]:
            flow_pred = flow_pred[:, : flow_latent.shape[1]]
        if self.use_classifier_free_guidance:
            uncond, text = flow_pred.chunk(2)
            flow_pred = uncond + self.guidance_scale * (text - uncond)
        _check_finite(f"TriTSP/_predict_flow/t={timestep}", flow_pred, self.raise_on_nan)
        return flow_pred

    def _predict_flow(self, flow_latent: torch.Tensor, timestep: int) -> torch.Tensor:
        return self._predict_flow_impl(flow_latent, timestep)

    def predict_noise(self, noisy_latent: torch.Tensor, timestep: int) -> torch.Tensor:
        noisy_latent = noisy_latent.to(device=self.device, dtype=self.dtype)
        snr = self.get_timestep_snr(int(timestep)).to(self.device)
        sigma = self._sigma_from_timestep(int(timestep))
        flow_to_ddpm = ddpm_to_flow_factor(snr).to(device=self.device)
        flow_latent = noisy_latent.float() / flow_to_ddpm.float()
        flow_pred = self._predict_flow_impl(flow_latent.to(dtype=self.dtype), int(timestep))
        x0_hat = flow_latent.float() - sigma * flow_pred.float()
        alpha, beta = get_alpha_prod_and_beta_prod(snr.float())
        eps = (noisy_latent.float() - torch.sqrt(alpha) * x0_hat) / torch.sqrt(torch.clamp(beta, min=1e-8))
        eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
        _check_finite(f"TriTSP/predict_noise/eps/t={timestep}", eps, self.raise_on_nan)
        return eps.to(dtype=noisy_latent.dtype)


    def decode_latent_for_loss(self, latent: torch.Tensor) -> torch.Tensor:
        """Differentiable VAE decode for small latent crops used by patch quality loss."""
        _check_finite("DuPCmdc/decode_latent_for_loss/input", latent, self.raise_on_nan)
        latent_for_vae = latent.to(device=self.device, dtype=self.text_vae_dtype)
        decoded = self.vae.decode(latent_for_vae / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)
        _check_finite("DuPCmdc/decode_latent_for_loss/output", image, self.raise_on_nan)
        return image

    def trainable_parameters(self, condition_mode: str = "full"):
        stats = self.tritsp_adapter.configure_trainable_for_mode(condition_mode)
        self.tritsp_train_condition_mode = self.normalize_condition_mode(condition_mode)
        print(
            f"[DuPCmdc] train_condition_mode={self.tritsp_train_condition_mode}; "
            f"trainable_adapter_params={stats['trainable']:,}/{stats['total']:,}"
        )


        return [p for p in self.tritsp_adapter.parameters() if p.requires_grad]

    def set_lowrank_trainable(self, enabled: bool, condition_mode: Optional[str] = None) -> None:
        self.tritsp_adapter.set_lowrank_trainable(enabled, condition_mode)

    def set_branch_logging(self, enabled: bool) -> None:
        self.tritsp_adapter.set_branch_logging(enabled)

    def reset_branch_contribution_stats(self) -> None:
        self.tritsp_adapter.reset_branch_contribution_stats()

    def consume_branch_contribution_stats(self, reset: bool = True):
        return self.tritsp_adapter.consume_branch_contribution_stats(reset=reset)

    def gate_regularization(self, condition_mode: str = "full") -> torch.Tensor:
        return self.tritsp_adapter.gate_regularization(condition_mode)

    def delta_regularization(self, condition_mode: str = "full") -> torch.Tensor:
        return self.tritsp_adapter.delta_regularization(condition_mode)

    def get_last_aux(self) -> Dict[str, float]:
        return self.tritsp_adapter.get_last_aux()

    def save_checkpoint(self, path: str | Path, extra: Optional[Dict[str, Any]] = None, optimizer=None, scheduler=None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "format": "DuPCmdc unified checkpoint",
            "architecture_version": self.tritsp_adapter.ARCHITECTURE_VERSION,
            "tritsp_adapter": self.tritsp_adapter.state_dict(),
            "tritsp_inject_block_indices": self.tritsp_inject_block_indices,
            "query_split": {
                "total": int(self.tritsp_adapter.num_condition_tokens),
                "tsp": int(self.tritsp_adapter.num_tsp_condition_tokens),
                "spp": int(self.tritsp_adapter.num_spp_condition_tokens),
            },
            "extra": extra or {},
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        torch.save(payload, p)

    def load_checkpoint(self, path: str | Path, optimizer=None, scheduler=None) -> Dict[str, Any]:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise RuntimeError(f"Unsupported checkpoint format: {path}")
        saved_arch = ckpt.get("architecture_version")
        compatible = {self.tritsp_adapter.ARCHITECTURE_VERSION,
                      "xiaorong_split_prior_component_isolation_v3",
                      "xiaorong_split_prior_decoupled_v2"}
        extra = ckpt.get("extra") or {}
        saved_mode = str(extra.get("train_condition_mode", (extra.get("args") or {}).get("train_condition_mode", "full"))).lower()
        if saved_arch not in compatible or saved_mode != "full":
            raise RuntimeError(f"Checkpoint is incompatible with full DPMA: architecture={saved_arch}, mode={saved_mode}")
        state = ckpt.get("tritsp_adapter", ckpt)
        self.tritsp_adapter.load_state_dict(state, strict=True)
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"[DuP-CMDC] loaded {path}; architecture={saved_arch}")
        return dict(extra)
