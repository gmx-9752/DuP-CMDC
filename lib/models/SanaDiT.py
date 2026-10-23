import math
import contextlib
from typing import Union

import torch
from diffusers import SanaPipeline
from tqdm import tqdm

from lib.models.latent_noise_prediction_model import LatentNoisePredictionModel
from lib.diffc.utils.alpha_beta import get_alpha_prod_and_beta_prod


def _as_torch_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Convert config strings such as 'float16'/'bfloat16' to torch dtypes."""
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).lower().replace("torch.", "")
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return table[name]


def sigma_to_snr(sigma: torch.Tensor) -> torch.Tensor:
    """
    Convert Sana/flow sigma to the std-ratio SNR used by DiffC.

    Sana's flow path can be written as
        z_sigma = (1 - sigma) * x0 + sigma * eps.
    DiffC's DDPM utilities use alpha = snr^2 / (1 + snr^2), so snr must be
    the signal/noise standard-deviation ratio, not log-SNR.
    """
    sigma = torch.as_tensor(sigma, dtype=torch.float32)
    sigma = torch.clamp(sigma, min=1e-6, max=1.0)
    return (1.0 - sigma) / sigma


def ddpm_to_flow_factor(snr: torch.Tensor) -> torch.Tensor:
    """
    Return the scalar c such that
        x_ddpm = c * z_flow.

    The flow latent is unnormalised: (1-sigma)*x0 + sigma*eps.
    DiffC's DDPM latent is normalised to unit variance:
        sqrt(alpha)*x0 + sqrt(beta)*eps.
    With the same signal/noise ratio, x_ddpm = z_flow / sqrt((1-sigma)^2+sigma^2).
    """
    snr = torch.as_tensor(snr, dtype=torch.float32)
    sigma = 1.0 / (snr + 1.0)
    _, beta_prod = get_alpha_prod_and_beta_prod(snr)
    return torch.sqrt(torch.clamp(beta_prod, min=1e-12)) / torch.clamp(sigma, min=1e-6)


def _check_finite(name: str, x: torch.Tensor, raise_on_nan: bool = True) -> None:
    """Print a useful diagnostic before a NaN tensor is silently saved as a black image."""
    if torch.isfinite(x).all():
        return
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        msg = f"[SanaDiT][NONFINITE] {name}: all values are NaN/Inf, shape={tuple(x.shape)}"
    else:
        msg = (
            f"[SanaDiT][NONFINITE] {name}: shape={tuple(x.shape)}, "
            f"finite_min={finite.min().item():.6g}, finite_max={finite.max().item():.6g}, "
            f"nan={(torch.isnan(x)).sum().item()}, inf={(torch.isinf(x)).sum().item()}"
        )
    if raise_on_nan:
        raise FloatingPointError(msg)
    print(msg)


class SanaDiTModel(LatentNoisePredictionModel):
    """
    Adapter that makes a local Sana DiT diffusers checkpoint usable by DiffC.

    Important implementation detail:
    - DiffC's encoder still expects an epsilon prediction under a DDPM path, so
      predict_noise() converts Sana's flow prediction into DDPM epsilon.
    - For final reconstruction, converting flow prediction to epsilon at very
      small timesteps is numerically fragile.  Therefore this adapter also
      provides a native flow-Euler denoise() method.  evaluate.py should call it
      when available instead of lib.diffc.denoise.denoise().
    """

    def __init__(
        self,
        model_id: str = "./weight",
        device: str = "cuda",
        dtype: Union[str, torch.dtype] = torch.float16,
        transformer_dtype: Union[str, torch.dtype, None] = None,
        text_vae_dtype: Union[str, torch.dtype] = torch.bfloat16,
        local_files_only: bool = True,
        max_sequence_length: int = 300,
        raise_on_nan: bool = True,
        disable_cudnn_for_transformer: bool = True,
    ):
        self.device = device
        self.dtype = _as_torch_dtype(transformer_dtype if transformer_dtype is not None else dtype)
        self.text_vae_dtype = _as_torch_dtype(text_vae_dtype)
        self.max_sequence_length = int(max_sequence_length)
        self.raise_on_nan = bool(raise_on_nan)
        # 1920x1088 + fp16 时，Sana patch_embed 里的 Conv2d 有时会触发：
        # RuntimeError: GET was unable to find an engine to execute this computation
        # 禁用 cuDNN 后会走更保守的 fallback 路径，速度可能慢一点，但更稳。
        self.disable_cudnn_for_transformer = bool(disable_cudnn_for_transformer)

        # The Sana docs recommend fp16 for the transformer of the 600M model, but
        # text encoder and VAE must remain bf16 or fp32.  Load the pipeline in the
        # VAE/text dtype, then cast the transformer explicitly.
        self.pipe = SanaPipeline.from_pretrained(
            model_id,
            torch_dtype=self.text_vae_dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        self.pipe.transformer.to(dtype=self.dtype)
        self.pipe.text_encoder.to(dtype=self.text_vae_dtype)
        self.pipe.vae.to(dtype=self.text_vae_dtype)

        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler

        self.vae.eval()
        self.transformer.eval()
        self.pipe.text_encoder.eval()

        """
        try:
            self.vae.enable_tiling()
        except Exception:
            pass
        try:
            self.vae.enable_slicing()
        except Exception:
            pass
        """
        # Sana 使用的是 AutoencoderDC，不是 Stable Diffusion 的 AutoencoderKL。
        # AutoencoderDC 在当前 diffusers 版本中没有实现 tiled_encode/tiled_decode。
        # 如果调用 self.vae.enable_tiling()，后面 self.vae.encode(img_pt)
        # 会自动进入 tiled_encode，然后报错：
        # NotImplementedError: `tiled_encode` has not been implemented for AutoencoderDC.
        #
        # 所以这里必须显式关闭 VAE tiling/slicing。
        try:
            self.vae.disable_tiling()
        except Exception:
            if hasattr(self.vae, "use_tiling"):
                self.vae.use_tiling = False

        try:
            self.vae.disable_slicing()
        except Exception:
            if hasattr(self.vae, "use_slicing"):
                self.vae.use_slicing = False

        self.num_train_timesteps = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        self.guidance_scale = 0.0
        self.use_classifier_free_guidance = False
        self.image_width = None
        self.image_height = None
        self.prompt_embeds = None
        self.prompt_attention_mask = None

    def _sigma_from_timestep(self, timestep: int) -> torch.Tensor:
        """
        Sana/flow timestep 近似是 sigma * num_train_timesteps。

        注意：
        DiffC 的 encode.py 会从 current_timestep=1000 开始。
        如果这里让 t=1000 对应 sigma=1.0，则 SNR=(1-sigma)/sigma=0。
        原始 DiffC 的 P() 函数会除以 sqrt(alpha_prod_t)，
        而 alpha_prod_t 在 SNR=0 时正好等于 0，第一步 1000->889 就会产生 Inf/NaN。

        所以这里必须把 sigma 限制在 (0, 1) 开区间内。
        sigma_max=0.999 仍然非常接近纯噪声，但可以避免除以 0。
        """
        sigma = float(timestep) / float(self.num_train_timesteps)

        # 不要让 sigma 精确等于 1.0，否则 t=1000 时 SNR=0，会把 DiffC 的 P() 算炸。
        sigma_min = 1e-5
        sigma_max = 0.999

        sigma = max(sigma_min, min(sigma_max, sigma))
        return torch.tensor(sigma, device=self.device, dtype=torch.float32)

    def get_timestep_snr(self, timestep: int):
        if int(timestep) == 0:
            return torch.inf
        return sigma_to_snr(self._sigma_from_timestep(int(timestep)))

    @torch.no_grad()
    def image_to_latent(self, img_pt: torch.Tensor) -> torch.Tensor:
        """Encode [0,1] RGB image tensor to Sana's 32x-compressed latent."""
        if img_pt.dim() == 3:
            img_pt = img_pt.unsqueeze(0)
        img_pt = img_pt.to(device=self.device, dtype=self.text_vae_dtype)
        img_pt = img_pt * 2.0 - 1.0
        enc = self.vae.encode(img_pt, return_dict=False)[0]
        latent = enc * self.vae.config.scaling_factor
        _check_finite("image_to_latent", latent, self.raise_on_nan)
        return latent.to(dtype=self.dtype).detach()

    @torch.no_grad()
    def latent_to_image(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode Sana latent back to [0,1] RGB image tensor."""
        _check_finite("latent_to_image/input_latent", latent, self.raise_on_nan)
        latent = torch.nan_to_num(latent.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        latent = latent.to(device=self.device, dtype=self.text_vae_dtype)
        image = self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False)[0]
        _check_finite("latent_to_image/decoded", image, self.raise_on_nan)
        return (image.float() / 2.0 + 0.5).clamp(0, 1).detach()

    @torch.no_grad()
    def configure(self, prompt: str, prompt_guidance: float, image_width: int, image_height: int):
        """Cache prompt embeddings and image size for subsequent predict_noise calls."""
        self.guidance_scale = float(prompt_guidance)
        self.use_classifier_free_guidance = self.guidance_scale > 1.0
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.pipe.encode_prompt(
            prompt=prompt or "",
            do_classifier_free_guidance=self.use_classifier_free_guidance,
            negative_prompt="",
            device=torch.device(self.device),
            clean_caption=False,
            max_sequence_length=self.max_sequence_length,
            complex_human_instruction=False,
        )

        if self.use_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        self.prompt_embeds = prompt_embeds.to(self.device)
        self.prompt_attention_mask = prompt_attention_mask.to(self.device)

    @torch.no_grad()
    def _predict_flow(self, flow_latent: torch.Tensor, timestep: int) -> torch.Tensor:
        """Run the Sana transformer and return flow prediction v = eps - x0."""
        if self.prompt_embeds is None:
            raise RuntimeError("Call SanaDiTModel.configure(...) before predict_noise/denoise.")

        flow_latent = flow_latent.to(device=self.device, dtype=self.dtype).contiguous()
        latent_model_input = (
            torch.cat([flow_latent] * 2, dim=0)
            if self.use_classifier_free_guidance
            else flow_latent
        )

        # 关键 1：确保送进 Sana patch_embed Conv2d 的 latent 是 contiguous。
        # 1920x1088 -> latent shape 为 (1, 32, 34, 60)，这个非方形高分辨率 shape
        # 在 fp16 + cuDNN 下更容易触发 backend engine 问题。
        latent_model_input = latent_model_input.to(
            device=self.device,
            dtype=self.dtype,
        ).contiguous()

        # 关键 2：prompt embedding 也保持 device/dtype/contiguous 一致。
        prompt_embeds = self.prompt_embeds.to(
            device=self.device,
            dtype=self.dtype,
        ).contiguous()

        prompt_attention_mask = self.prompt_attention_mask.to(device=self.device)

        t = torch.full(
            (latent_model_input.shape[0],),
            float(timestep),
            device=self.device,
            dtype=torch.float32,
        )
        t = t * float(getattr(self.transformer.config, "timestep_scale", 1.0))

        # 关键 3：禁用 cuDNN，只包住 Sana transformer 这一次前向。
        # 这样不会改变 DiffC 数学公式，也不会改变 encoding_timesteps/manual_dkl_per_step；
        # 只是绕开 cuDNN 在某些 fp16 Conv2d shape 上找不到 engine 的问题。
        cudnn_ctx = (
            torch.backends.cudnn.flags(enabled=False)
            if self.disable_cudnn_for_transformer
            else contextlib.nullcontext()
        )

        with cudnn_ctx:
            flow_pred = self.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=t,
                return_dict=False,
            )[0].float()

        if flow_pred.shape[1] != flow_latent.shape[1]:
            flow_pred = flow_pred[:, : flow_latent.shape[1]]

        if self.use_classifier_free_guidance:
            flow_uncond, flow_text = flow_pred.chunk(2)
            flow_pred = flow_uncond + self.guidance_scale * (flow_text - flow_uncond)

        _check_finite(f"_predict_flow/t={timestep}", flow_pred, self.raise_on_nan)
        return flow_pred

    @torch.no_grad()
    def predict_noise(self, noisy_latent: torch.Tensor, timestep: int) -> torch.Tensor:
        """
        Return DDPM epsilon prediction required by DiffC encoding.

        This conversion is only used for P/Q during encoding and schedule search.
        Final reconstruction should use denoise(), which stays in Sana's native
        flow coordinates and is much more stable near t=0.
        """
        noisy_latent = noisy_latent.to(device=self.device, dtype=self.dtype)
        snr = self.get_timestep_snr(int(timestep)).to(self.device)
        sigma = self._sigma_from_timestep(int(timestep))
        flow_to_ddpm = ddpm_to_flow_factor(snr).to(device=self.device)
        flow_latent = noisy_latent.float() / flow_to_ddpm.float()

        _check_finite(f"predict_noise/input_noisy_latent/t={timestep}", noisy_latent, self.raise_on_nan)
        _check_finite(f"predict_noise/input_flow_latent/t={timestep}", flow_latent, self.raise_on_nan)

        flow_pred = self._predict_flow(flow_latent.to(dtype=self.dtype), int(timestep))

        # Diffusers' DPMSolver uses x0_pred = sample - sigma * flow_pred for flow_prediction.
        x0_hat = flow_latent.float() - sigma * flow_pred.float()
        alpha_prod_t, beta_prod_t = get_alpha_prod_and_beta_prod(snr.float())
        eps = (noisy_latent.float() - torch.sqrt(alpha_prod_t) * x0_hat) / torch.sqrt(
            torch.clamp(beta_prod_t, min=1e-8)
        )
        eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
        _check_finite(f"predict_noise/eps/t={timestep}", eps, self.raise_on_nan)
        return eps.to(dtype=noisy_latent.dtype)

    @torch.no_grad()
    def denoise(self, noisy_latent: torch.Tensor, latent_timestep: int, timestep_schedule) -> torch.Tensor:
        """
        Native Sana flow-Euler denoising for DiffC reconstructions.

        DiffC's original lib.diffc.denoise converts model output to DDPM epsilon
        and performs a DDIM update.  For Sana this is numerically fragile at the
        last few timesteps because beta is very small.  Here we denoise in flow
        coordinates instead:
            z_{sigma_next} = z_sigma + (sigma_next - sigma) * v_theta(z_sigma, sigma)
        where v_theta is Sana's flow_prediction.  At sigma=0 the flow latent is x0.
        """
        current_t = int(latent_timestep)
        snr = self.get_timestep_snr(current_t).to(self.device)
        flow_to_ddpm = ddpm_to_flow_factor(snr).to(device=self.device)
        flow_latent = noisy_latent.to(self.device).float() / flow_to_ddpm.float()
        current_sigma = self._sigma_from_timestep(current_t)

        schedule = [int(t) for t in timestep_schedule if int(t) < current_t]
        if len(schedule) == 0 or schedule[-1] != 0:
            schedule = schedule + [0]

        for next_t in tqdm(schedule):
            flow_pred = self._predict_flow(flow_latent.to(dtype=self.dtype), current_t).float()
            next_sigma = self._sigma_from_timestep(next_t) if next_t > 0 else torch.tensor(0.0, device=self.device)
            flow_latent = flow_latent + (next_sigma - current_sigma) * flow_pred
            flow_latent = torch.nan_to_num(flow_latent, nan=0.0, posinf=20.0, neginf=-20.0)
            current_t = int(next_t)
            current_sigma = next_sigma

        _check_finite("denoise/final_latent", flow_latent, self.raise_on_nan)
        return flow_latent.to(dtype=noisy_latent.dtype)
