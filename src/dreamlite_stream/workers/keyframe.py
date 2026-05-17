"""KeyframeWorker — text-to-image worker that produces a stable anchor latent.

Run once at scene start. The TE here is the [Generate] mode (text-only Qwen3-VL
without vision tokens), so the resulting prompt embedding is image-INDEPENDENT
and safe to cache forever — unlike the [Edit] mode used by EditWorker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from PIL import Image

from .. import pipeline_ops as ops
from ..state import SharedState


@dataclass
class KeyframeWorker:
    pipeline: object
    state: SharedState
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    seed: Optional[int] = 42

    @torch.no_grad()
    def generate(self, prompt: Optional[str] = None) -> Tuple[torch.Tensor, Image.Image]:
        """Generate a keyframe latent from text and store it in SharedState.

        Returns (keyframe_latent, keyframe_image_pil) for inspection. The
        latent is also written to `state.keyframe_latent` and the cached gen
        prompt embeddings are stored in `state.cached_gen_prompt_embeds/mask`
        so subsequent EditWorker calls can reuse them if they ever switch to
        generate-mode TE.
        """
        s = self.state
        text = prompt if prompt is not None else s.prompt

        # Reset and configure the scheduler for this resolution / step count.
        timesteps, _ = ops.set_timesteps(
            self.pipeline,
            height=s.height, width=s.width,
            num_inference_steps=s.num_inference_steps,
            device=self.device,
        )
        time_ids = ops.make_time_ids(s.width, s.height, self.device, self.dtype)

        # Text-only TE — image-independent, safe to cache.
        prompt_embeds, mask = ops.encode_prompt_generate(
            self.pipeline, text, self.device, self.dtype,
        )
        s.cached_gen_prompt_embeds = prompt_embeds
        s.cached_gen_prompt_mask = mask

        gen = None
        if self.seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
        init = ops.make_init_noise(
            self.pipeline,
            height=s.height, width=s.width,
            device=self.device, dtype=self.dtype, generator=gen,
        )
        # In generate mode the official pipeline uses zeros for the cond image.
        zero_cond = torch.zeros_like(init)

        latent = ops.denoise(
            self.pipeline, init, zero_cond, prompt_embeds, mask,
            time_ids, timesteps,
        )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        s.keyframe_latent = latent
        images = ops.decode_latent(self.pipeline, latent, output_type="pil")
        return latent, images[0]
