"""Post-hoc output blending for temporal smoothing.

Operates entirely on decoded RGB frames — does NOT touch the diffusion
model's conditioning, so it preserves the model's training distribution
(the failure mode that broke MVP-2 cond manipulation).

Mechanism:
    out_t = alpha * decoded_t + (1 - alpha) * warp(out_{t-1}, flow)

  alpha = 1.0 -> no blending (= MVP-1 baseline)
  alpha = 0.7 -> 30% drag from previous output (gentle smoothing)
  alpha = 0.5 -> 50/50 blend (heavier smoothing; risk of blur)

Flow is computed on INPUT frames (Farneback) — same convention as the
metrics module, so warping_error and the blend mechanism agree on
"what counts as motion".

Cost: ~5-10ms (Farneback) + remap + alpha blend per frame at 512x512.
Roughly 15-20ms total per frame on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import flow as flowlib


@dataclass
class OutputBlender:
    alpha: float = 1.0  # 1.0 = disabled
    _prev_out: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _prev_in_gray: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._prev_out = None
        self._prev_in_gray = None

    def apply(self, decoded_rgb: np.ndarray, input_rgb: np.ndarray) -> np.ndarray:
        """Return blended output frame; remembers it for the next call."""
        in_gray = flowlib.to_gray(input_rgb)
        if self.alpha >= 1.0 or self._prev_out is None or self._prev_in_gray is None:
            self._prev_out = decoded_rgb
            self._prev_in_gray = in_gray
            return decoded_rgb

        flow = flowlib.farneback_flow(self._prev_in_gray, in_gray)
        H, W = flow.shape[:2]
        xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        # Forward warp prev_out to curr-frame coords: gather with MINUS flow.
        map_x = xs - flow[..., 0]
        map_y = ys - flow[..., 1]
        warped_prev = cv2.remap(
            self._prev_out, map_x, map_y,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        blended = (
            self.alpha * decoded_rgb.astype(np.float32)
            + (1.0 - self.alpha) * warped_prev.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        self._prev_out = blended
        self._prev_in_gray = in_gray
        return blended
