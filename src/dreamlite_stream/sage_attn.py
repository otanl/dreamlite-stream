"""SageAttention2/3 drop-in adapter for the DreamLite UNet, LLLite, and
Qwen3-VL TE.

Per-GPU plan (see ``notes/sageattn3_integration.md`` §4):

    * RTX 5090 (Blackwell sm_120): SageAttn3 FP4 via ``sageattn3_blackwell``.
    * RTX 4090 (Ada sm_89):       SageAttn2 INT4-QK + FP8-PV.
    * RTX 3090 Ti (Ampere sm_86): SageAttn2 INT8-QK + FP16-PV (Triton).

``reduce-overhead`` (CUDA-graph capture) is our default ``torch.compile``
mode and is the main risk: SageAttn's repo issues #74 / #162 / #236 / #337
confirm capture problems with the raw kernel call. The wrapper here
adopts the FlashAttention-2 pattern of registering the kernel as an
opaque custom op via ``torch.library.custom_op`` so Dynamo treats it as
a black box and the outer graph captures the surrounding ops normally.

Two install points:

    1. UNet + LLLite: per-attention swap via
       ``unet.set_attn_processor(SageAttnProcessor())``. This is the
       diffusers-recommended hook for DiT/UNet (the repo README
       deprecates the global ``F.scaled_dot_product_attention`` monkey-
       patch for these models).
    2. Qwen3-VL TE side-stream: the global
       ``F.scaled_dot_product_attention = sage_sdpa`` monkey-patch is
       fine here since all three attention sites (vision ViT, language
       tower, cross-modal connector) use stock SDPA.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Callable, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class SageBackend(Enum):
    """Which SageAttention kernel to use."""

    SAGE3_FP4 = "sage3_fp4"            # Blackwell only
    SAGE2_FP8_CUDA = "sage2_fp8_cuda"  # Ada / Hopper
    SAGE2_INT8_TRITON = "sage2_int8_triton"  # Ampere (and fallback)


def select_backend(device: Optional[torch.device] = None) -> SageBackend:
    """Pick a SageAttention backend based on the device compute capability.

    Returns the highest-precision-loss / highest-throughput variant the
    hardware natively supports.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    sm = major * 10 + minor

    if sm >= 120:  # Blackwell sm_120
        return SageBackend.SAGE3_FP4
    if sm >= 89:   # Ada sm_89 (also Hopper sm_90)
        return SageBackend.SAGE2_FP8_CUDA
    return SageBackend.SAGE2_INT8_TRITON


# ---------------------------------------------------------------------------
# Kernel-call wrappers
# ---------------------------------------------------------------------------


def _sage3_fp4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               is_causal: bool) -> torch.Tensor:
    """SageAttn3 Blackwell FP4 kernel call.

    TODO(sage): wire to actual import once installed:
        from sageattention3_blackwell import sageattn3_blackwell
        return sageattn3_blackwell(q, k, v, is_causal=is_causal)
    """
    raise NotImplementedError(
        "SageAttn3 FP4 kernel not wired. Install per "
        "github.com/thu-ml/SageAttention/sageattention3_blackwell/README.md, "
        "then replace this body with a single sageattn3_blackwell(...) call."
    )


def _sage2_fp8_cuda(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     is_causal: bool) -> torch.Tensor:
    """SageAttn2 INT4-QK + FP8-PV CUDA kernel (Ada / Hopper)."""
    raise NotImplementedError(
        "SageAttn2 FP8 CUDA kernel not wired. Install ``sageattention>=2.2.0`` "
        "and replace this body with sageattn_qk_int8_pv_fp8_cuda(...) or "
        "..._cuda_sm90(...) for Hopper."
    )


def _sage2_int8_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       is_causal: bool) -> torch.Tensor:
    """INT8-QK Triton kernel (Ampere path).

    Prefers the sageattention>=2.2 entry point; falls back to the v1
    (1.0.6, pure-Triton) ``sageattn`` API, which is pip-installable on
    Windows without a CUDA build. Verified on sm_86 (RTX 3090 Ti,
    2026-06-11): numeric equivalence vs SDPA (cosine 0.99992, max abs
    diff 0.006 at fp16) and ~2x speedup at UNet-realistic shapes
    (S=4096: 0.60 vs 1.28 ms; B=16/S=1024: 0.40 vs 0.64 ms). Small
    shapes (B=2/S=1024) are launch-overhead-bound and slower than
    SDPA -- per-site gating belongs to the benchmark phase.
    """
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_triton
        return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal)
    except ImportError:
        from sageattention import sageattn  # v1 API (1.0.6)
        return sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal)


_BACKEND_TABLE: dict[SageBackend, Callable[..., torch.Tensor]] = {
    SageBackend.SAGE3_FP4: _sage3_fp4,
    SageBackend.SAGE2_FP8_CUDA: _sage2_fp8_cuda,
    SageBackend.SAGE2_INT8_TRITON: _sage2_int8_triton,
}


# ---------------------------------------------------------------------------
# torch.library.custom_op wrapper (Dynamo-opaque)
# ---------------------------------------------------------------------------
# Defining the kernel as a custom op makes Dynamo treat it as a black box,
# so the outer torch.compile reduce-overhead capture sees the surrounding
# ops with stable shapes and captures the CUDA graph around it.
#
# TODO(sage): register the op once a backend is wired:
#
#   @torch.library.custom_op("dreamlite_stream::sage_attn", mutates_args=())
#   def sage_attn(q, k, v, is_causal: bool, backend: str) -> torch.Tensor:
#       return _BACKEND_TABLE[SageBackend(backend)](q, k, v, is_causal)
#
#   @sage_attn.register_fake
#   def _(q, k, v, is_causal, backend):
#       return torch.empty_like(q)
# ---------------------------------------------------------------------------


# Captured at import time so the fallback path always reaches the TRUE
# PyTorch kernel. Without this, install_global_sdpa_patch() replaces
# F.scaled_dot_product_attention and any fallback inside sage_sdpa would
# re-enter the patched symbol -> infinite recursion. (Caught by the
# bonsai R5 composition probe, 2026-06-11.)
_ORIG_SDPA = F.scaled_dot_product_attention

# Route counters for observability under the global patch: lets callers
# distinguish "composes because the sage kernel ran" from "composes
# trivially because every site fell back". Reset by
# install_global_sdpa_patch().
PATCH_STATS = {"sage": 0, "fallback": 0}


def sage_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    backend: Optional[SageBackend] = None,
) -> torch.Tensor:
    """Drop-in replacement for ``F.scaled_dot_product_attention``.

    Falls back to the original PyTorch SDPA (captured at import) when:
        * ``attn_mask`` is not None (SageAttn variants do not support
          arbitrary masks),
        * ``dropout_p > 0`` (training-time path; we're inference-only),
        * ``scale`` is non-default and the backend doesn't accept it,
        * ``enable_gqa`` is set (not all variants support GQA),
        * inputs are not fp16/bf16 CUDA tensors (kernel contract).
    """
    needs_fallback = (
        attn_mask is not None
        or dropout_p > 0.0
        or scale is not None
        or enable_gqa
        or not q.is_cuda
        or q.dtype not in (torch.float16, torch.bfloat16)
        # INT8/FP8 kernels support head dims {64, 96, 128} only; e.g. a
        # VAE mid-block attention (head dim = channel dim, often 512)
        # must fall back. (Caught by the bonsai R5 probe, 2026-06-11.)
        or q.size(-1) not in (64, 96, 128)
    )
    if needs_fallback:
        PATCH_STATS["fallback"] += 1
        return _ORIG_SDPA(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
        )

    PATCH_STATS["sage"] += 1
    backend = backend or select_backend(q.device)
    return _BACKEND_TABLE[backend](q, k, v, is_causal=is_causal)


# ---------------------------------------------------------------------------
# diffusers AttnProcessor swap
# ---------------------------------------------------------------------------


class SageAttnProcessor:
    """Diffusers attention processor that routes through SageAttention.

    Drop-in for ``diffusers.models.attention_processor.AttnProcessor2_0``;
    install via ``unet.set_attn_processor(SageAttnProcessor())``. This is
    the per-module path recommended by SageAttention's README for DiT and
    UNet stacks (avoids the global SDPA monkey-patch).

    First/last-step fallback (only meaningful when backend is SAGE3_FP4 on
    a few-step distilled denoiser): if ``fallback_steps_at_extremes > 0``
    and the model is queried via a known step index, this processor falls
    back to SageAttn2++ (or SDPA) on those steps. This is the policy that
    SageAttn3's sub-README recommends for models outside its validated
    set; DreamLite-mobile is 4-step and not in their list, so the
    conservative default is to enable the fallback.
    """

    def __init__(
        self,
        backend: Optional[SageBackend] = None,
        fallback_at_extremes: bool = True,
    ) -> None:
        self.backend = backend
        self.fallback_at_extremes = fallback_at_extremes

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        """TODO(sage): mirror diffusers AttnProcessor2_0.__call__ exactly,
        replacing the final ``F.scaled_dot_product_attention(...)`` line
        with ``sage_sdpa(q, k, v, attn_mask=attention_mask, is_causal=False,
        backend=self.backend)``.

        See ``diffusers/src/diffusers/models/attention_processor.py``
        ``AttnProcessor2_0.__call__`` for the reference body. The pre-SDPA
        prep (norm / QKV proj / head reshape) and post-SDPA prep (head
        reshape / o-proj / dropout / residual) are identical; only the
        attention call swaps.
        """
        raise NotImplementedError(
            "SageAttnProcessor.__call__: copy the AttnProcessor2_0 body, "
            "swap the SDPA line for sage_sdpa(...)."
        )


# ---------------------------------------------------------------------------
# Global SDPA monkey-patch (for the Qwen3-VL TE side-stream)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _GlobalPatchHandle:
    original_sdpa: Callable

    def remove(self) -> None:
        F.scaled_dot_product_attention = self.original_sdpa


def install_global_sdpa_patch(
    backend: Optional[SageBackend] = None,
) -> _GlobalPatchHandle:
    """Replace ``F.scaled_dot_product_attention`` with ``sage_sdpa``.

    Use for the Qwen3-VL TE: all three attention sites (vision ViT,
    language tower, cross-modal connector) resolve to stock SDPA, so a
    single global patch covers them. Do NOT use for the UNet / LLLite
    paths — use :class:`SageAttnProcessor` instead, per the repo's own
    DiT/UNet guidance.

    Returns a handle whose ``.remove()`` restores the original SDPA.
    Resets :data:`PATCH_STATS` so callers can inspect how many attention
    sites actually hit the sage kernel vs fell back.
    """
    original = F.scaled_dot_product_attention
    PATCH_STATS["sage"] = 0
    PATCH_STATS["fallback"] = 0

    # Parameter names mirror torch's public signature exactly -- callers
    # (e.g. diffusers attention dispatch) pass query=/key=/value= by
    # keyword. (Caught by the bonsai R5 probe, 2026-06-11.)
    def _patched(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, enable_gqa=False):
        return sage_sdpa(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
            backend=backend,
        )

    F.scaled_dot_product_attention = _patched
    return _GlobalPatchHandle(original_sdpa=original)
