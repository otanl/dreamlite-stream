"""Distillation loss family for the follow-up paper.

Companion to the LCM-LoRA training script (`scripts/train_lcm_lora.py`).
Adds three optional terms on top of the standard pixel/latent MSE loss
so the smoothing artifact documented in our paper (§V LCM-LoRA
case study) is prevented at training time rather than only diagnosed
post-hoc.

Loss family (see `followup_design.md` §3.1):

    L = alpha * L_pixel + beta * L_perc + gamma * L_spec + delta * L_adv

Each term is a separate function so the training script can disable any
of them by setting its coefficient to zero. The default configuration
(O3 in the design doc) is alpha=1.0, beta=0.1, gamma=0.05, delta=0.0;
that is, MSE + perceptual + spectral, no adversarial.

Implementation notes:

- All operations are differentiable PyTorch ops on GPU tensors.
- `L_perc` (LPIPS) operates on RGB pixel tensors in [-1, 1] of shape
  (N, 3, H, W); the caller is expected to have decoded student and
  teacher latents through the VAE before calling. We DO NOT decode
  inside the loss to give the caller control over compute/memory
  trade-offs (decode on every step is heavy; the caller can sample
  every k-th step instead).
- `L_spec` (HF-FFT log-ratio) can operate either in latent or pixel
  space; the design defaults to pixel space because the smoothing
  artifact is fundamentally a pixel-space phenomenon, but the latent
  path is supported as a cheaper proxy.
- `L_adv` is intentionally NOT implemented in this initial module — it
  is a contingency (O4) and will be added if the perceptual+spectral
  pair does not close the sharpness gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class DistillationLossConfig:
    """Coefficients for the loss family. See `followup_design.md` §3.1."""

    alpha_pixel: float = 1.0      # standard v-prediction MSE
    beta_perc: float = 0.0        # LPIPS perceptual (off by default; enable for O1/O3)
    gamma_spec: float = 0.0       # HF-FFT log-ratio (off by default; enable for O2/O3)
    delta_adv: float = 0.0        # adversarial (off by default; contingency O4)

    # When True, compute L_spec on RGB pixels (requires VAE decode).
    # When False, compute L_spec on latents directly (cheaper, less accurate).
    spec_on_pixel: bool = True

    # When non-zero, only compute the heavy perceptual / pixel-spectral
    # terms every k-th training step. Set to 0 to compute every step.
    # Latent-MSE always runs every step.
    heavy_term_every_k_steps: int = 0

    @property
    def needs_pixel_decode(self) -> bool:
        """True if any active term needs the student / teacher pixels.

        L_perc always needs RGB pixels; L_spec needs them only when
        `spec_on_pixel` is True; L_adv needs them. L_pixel does not.
        """
        if self.beta_perc > 0.0:
            return True
        if self.gamma_spec > 0.0 and self.spec_on_pixel:
            return True
        if self.delta_adv > 0.0:
            return True
        return False


# --------------------------------------------------------------------------- #
# Pixel MSE (standard)
# --------------------------------------------------------------------------- #


def pixel_mse_loss(
    student_v_pred: torch.Tensor,
    target_v: torch.Tensor,
) -> torch.Tensor:
    """Standard v-prediction MSE in latent space.

    This is the baseline LCM-LoRA loss. Kept here as a function so the
    full-loss combiner can call it uniformly with the other terms.
    """
    return F.mse_loss(student_v_pred.float(), target_v.float())


# --------------------------------------------------------------------------- #
# LPIPS perceptual loss
# --------------------------------------------------------------------------- #


class _LPIPSWrapper:
    """Lazy-loaded LPIPS network, AlexNet backbone (light variant)."""

    _net = None

    @classmethod
    def get(cls, device: torch.device) -> "lpips.LPIPS":  # type: ignore[name-defined]
        if cls._net is None:
            import lpips  # imported lazily — only required when L_perc is enabled

            cls._net = lpips.LPIPS(net="alex", verbose=False)
            cls._net.eval()
            for p in cls._net.parameters():
                p.requires_grad_(False)
        if cls._net.parameters().__next__().device != device:
            cls._net = cls._net.to(device)
        return cls._net


def perceptual_lpips_loss(
    student_pixels: torch.Tensor,
    teacher_pixels: torch.Tensor,
) -> torch.Tensor:
    """LPIPS distance (AlexNet) between student and teacher pixel batches.

    Both tensors are expected to be RGB in the [-1, 1] range with shape
    (N, 3, H, W). Caller is responsible for VAE-decoding the latents
    before invoking this function (see `DistillationLossConfig` docs).
    """
    assert student_pixels.shape == teacher_pixels.shape, (
        f"LPIPS inputs must match shapes; got {student_pixels.shape} vs "
        f"{teacher_pixels.shape}"
    )
    assert student_pixels.size(1) == 3, "LPIPS expects 3-channel RGB input"
    net = _LPIPSWrapper.get(student_pixels.device)
    # lpips expects input in [-1, 1]; verify range loosely.
    return net(student_pixels, teacher_pixels).mean()


# --------------------------------------------------------------------------- #
# HF-FFT spectral loss
# --------------------------------------------------------------------------- #


def _hf_mask(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Boolean mask of FFT bins outside the center disc of radius min(h, w)/4."""
    cy, cx = h // 2, w // 2
    y = torch.arange(h, device=device).view(-1, 1)
    x = torch.arange(w, device=device).view(1, -1)
    r2 = (y - cy) ** 2 + (x - cx) ** 2
    return r2 > (min(h, w) // 4) ** 2


def hf_fft_log_ratio_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """High-frequency FFT energy log-ratio loss.

    L_spec = |log HF(student) - log HF(teacher)|

    where HF(x) is the mean magnitude of FFT2(x) outside the center disc
    of radius H/4 (matching the eval probe in `metrics.hf_density`).

    Operates on whatever input is given (latent or pixel); the caller
    controls which via `DistillationLossConfig.spec_on_pixel`.

    Inputs:
        student, teacher: tensors of shape (N, C, H, W). Grayscale
        conversion is done by averaging over channels (consistent with
        the cv2 grayscale path in metrics.hf_density).
    """
    assert student.shape == teacher.shape, (
        f"HF-FFT inputs must match shapes; got {student.shape} vs "
        f"{teacher.shape}"
    )
    if student.dim() != 4:
        raise ValueError(f"expected 4D input, got shape {student.shape}")

    # Collapse to single-channel (mean across channels, mirrors grayscale).
    s = student.float().mean(dim=1)   # (N, H, W)
    t = teacher.float().mean(dim=1)

    s_fft = torch.fft.fft2(s)
    t_fft = torch.fft.fft2(t)
    s_fft = torch.fft.fftshift(s_fft, dim=(-2, -1))
    t_fft = torch.fft.fftshift(t_fft, dim=(-2, -1))

    s_mag = s_fft.abs()
    t_mag = t_fft.abs()

    n, h, w = s_mag.shape
    mask = _hf_mask(h, w, s_mag.device)  # (H, W) boolean
    # Apply mask per-image and compute mean magnitude.
    s_hf = s_mag[:, mask].mean(dim=-1)   # (N,)
    t_hf = t_mag[:, mask].mean(dim=-1)

    # Detach teacher so the loss only flows through the student.
    t_hf = t_hf.detach()

    return torch.abs(torch.log(s_hf + eps) - torch.log(t_hf + eps)).mean()


# --------------------------------------------------------------------------- #
# Combined loss
# --------------------------------------------------------------------------- #


@dataclass
class DistillationLossOutput:
    """Returned by `compute_distillation_loss`. All scalars are float CPU values."""

    total: torch.Tensor               # the loss to backprop
    pixel: float                       # MSE component (always)
    perc: Optional[float] = None       # LPIPS component (if enabled)
    spec: Optional[float] = None       # HF-FFT component (if enabled)


def compute_distillation_loss(
    config: DistillationLossConfig,
    *,
    # MSE inputs (latent space, always required).
    student_v_pred: torch.Tensor,
    target_v: torch.Tensor,
    # Optional pixel inputs (only required if config.needs_pixel_decode is True).
    student_pixels: Optional[torch.Tensor] = None,
    teacher_pixels: Optional[torch.Tensor] = None,
    # Optional latent inputs (only required if gamma_spec > 0 and not spec_on_pixel).
    student_latents: Optional[torch.Tensor] = None,
    teacher_latents: Optional[torch.Tensor] = None,
    # Optional step counter for `heavy_term_every_k_steps` gating.
    step: Optional[int] = None,
) -> DistillationLossOutput:
    """Compose the total distillation loss from active terms.

    The caller decides when to decode latents to pixels and pass them
    in (None means the perceptual / pixel-spectral terms are skipped
    for this step, even if their coefficient is non-zero). This keeps
    the heavy compute under the caller's control.
    """
    # Always compute pixel MSE in latent space.
    loss_pixel = pixel_mse_loss(student_v_pred, target_v)
    total = config.alpha_pixel * loss_pixel

    out = DistillationLossOutput(total=total, pixel=loss_pixel.item())

    heavy_enabled = (
        config.heavy_term_every_k_steps == 0
        or step is None
        or step % config.heavy_term_every_k_steps == 0
    )

    if config.beta_perc > 0.0 and heavy_enabled:
        if student_pixels is None or teacher_pixels is None:
            raise ValueError(
                "perceptual loss requested but student/teacher pixels not provided"
            )
        loss_perc = perceptual_lpips_loss(student_pixels, teacher_pixels)
        total = total + config.beta_perc * loss_perc
        out.perc = loss_perc.item()

    if config.gamma_spec > 0.0 and heavy_enabled:
        if config.spec_on_pixel:
            if student_pixels is None or teacher_pixels is None:
                raise ValueError(
                    "pixel-space HF-FFT loss requested but student/teacher pixels "
                    "not provided"
                )
            spec_s, spec_t = student_pixels, teacher_pixels
        else:
            if student_latents is None or teacher_latents is None:
                raise ValueError(
                    "latent-space HF-FFT loss requested but student/teacher latents "
                    "not provided"
                )
            spec_s, spec_t = student_latents, teacher_latents
        loss_spec = hf_fft_log_ratio_loss(spec_s, spec_t)
        total = total + config.gamma_spec * loss_spec
        out.spec = loss_spec.item()

    if config.delta_adv > 0.0:
        raise NotImplementedError(
            "Adversarial loss (L_adv, contingency O4) is intentionally not "
            "implemented in this initial module. See followup_design.md "
            "§3.1 for the design notes; add the discriminator + GAN training "
            "scaffolding here if the perceptual+spectral pair fails to close "
            "the sharpness gap."
        )

    out.total = total
    return out


# --------------------------------------------------------------------------- #
# Loss-config presets (match the design doc)
# --------------------------------------------------------------------------- #


def preset_B3_baseline() -> DistillationLossConfig:
    """B3 baseline: pure v-prediction MSE, unblended teacher target (rank 32 was
    the model side). The paper's §V LCM-LoRA v3 reference."""
    return DistillationLossConfig(alpha_pixel=1.0, beta_perc=0.0, gamma_spec=0.0)


def preset_O1_perceptual() -> DistillationLossConfig:
    """O1: pixel + perceptual."""
    return DistillationLossConfig(alpha_pixel=1.0, beta_perc=0.1, gamma_spec=0.0)


def preset_O2_spectral() -> DistillationLossConfig:
    """O2: pixel + spectral (HF-FFT on pixel space)."""
    return DistillationLossConfig(alpha_pixel=1.0, beta_perc=0.0, gamma_spec=0.05)


def preset_O3_main() -> DistillationLossConfig:
    """O3: pixel + perceptual + spectral. MAIN PROPOSED CONFIGURATION."""
    return DistillationLossConfig(
        alpha_pixel=1.0, beta_perc=0.1, gamma_spec=0.05, delta_adv=0.0
    )
