"""Shared state passed between workers in the streaming runtime.

For MVP-1 there is only one worker (EditWorker) so most fields exist as
forward-looking scaffolding. Future workers (keyframe / speculative /
repair) will read and write the same SharedState.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import torch


@dataclass
class SharedState:
    height: int = 512
    width: int = 512
    num_inference_steps: int = 4

    prompt: str = ""

    # Cached prompt embedding for [Generate] mode. Image-independent so it
    # can be reused across frames when the prompt is fixed. Edit-mode embeds
    # are NOT cached here because they depend on the per-frame ref image.
    cached_gen_prompt_embeds: Optional[torch.Tensor] = None
    cached_gen_prompt_mask: Optional[torch.Tensor] = None

    # Cached timesteps for the current (H, W, num_inference_steps) tuple.
    cached_timesteps: Optional[torch.Tensor] = None
    cached_time_ids: Optional[torch.Tensor] = None

    # Anchor frame chosen at scene start. Future use: keyframe worker writes
    # this; edit worker reads it as a long-term reference.
    keyframe_latent: Optional[torch.Tensor] = None

    # Most recent generated latents (newest at the right end).
    prev_latents: Deque[torch.Tensor] = field(default_factory=lambda: deque(maxlen=4))

    # Most recent input frame in grayscale (for optical flow on next frame).
    # numpy uint8 (H, W).
    prev_input_gray: Optional["object"] = None  # Optional[np.ndarray] avoiding hard import

    # Most recent DECODED output frame (RGB uint8 (H, W, 3)). Used by the
    # temporal LLLite path to provide the "warped previous output" cond.
    prev_decoded_rgb: Optional["object"] = None

    frame_idx: int = 0

    def push_latent(self, latent: torch.Tensor) -> None:
        self.prev_latents.append(latent)
        self.frame_idx += 1

    def last_latent(self) -> Optional[torch.Tensor]:
        return self.prev_latents[-1] if self.prev_latents else None

    def reset(self) -> None:
        self.prev_latents.clear()
        self.keyframe_latent = None
        self.prev_input_gray = None
        self.prev_decoded_rgb = None
        self.frame_idx = 0
