"""SpeculativeEditWorker — wraps an EditWorker with adaptive frame skipping.

Verifier policy (cheap, network-free):
  - Compute flow magnitude on inputs (prev_in_gray, curr_in_gray) via Farneback.
  - If max flow magnitude < `flow_thresh` AND we haven't speculated
    `max_consec` times in a row, ACCEPT speculation:
        latent_curr = warp_latent(prev_latent, flow)
        out         = VAE.decode(latent_curr)
    (no UNet denoise, no LLLite, no TE — just flow + warp + decode)
  - Otherwise REJECT and run a full edit step (which goes through the wrapped
    EditWorker, optionally LLLite-augmented). Reset the consec counter.

The prerequisite for a useful hit rate is that the wrapped worker produces a
TEMPORALLY CONSISTENT output stream — empirically shown by the temporal LLLite
adapter. With flickery base the speculation L1 is at the noise floor and no
hit threshold separates accept from reject.

Cost model per frame (steady state):
  hit:    flow ~10ms + warp ~5ms + VAE_dec ~30ms       = ~45ms  ( ~22 FPS)
  miss:   wrapped worker.step()                         = ~750ms with LLLite
  expected = (1 - h) * miss + h * hit
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .. import flow as flowlib
from .. import pipeline_ops as ops
from .edit import EditWorker, StepTiming


@dataclass
class SpecTiming(StepTiming):
    accepted: bool = False
    flow_max: float = 0.0


@dataclass
class SpeculativeEditWorker:
    """Composes around an EditWorker. Don't pre-compile its inner UNet — the
    speculative path skips UNet entirely on hit, so saving compile cost only
    helps the miss-path latency."""

    inner: EditWorker
    flow_thresh: float = 20.0   # px; max flow magnitude tolerated before forcing a miss
    max_consec: int = 4         # force a miss after this many consecutive hits (drift bound)
    _prev_input_gray: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _prev_latent: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _consec_hits: int = field(default=0, init=False, repr=False)

    @property
    def device(self) -> str:
        return self.inner.device

    @property
    def dtype(self) -> torch.dtype:
        return self.inner.dtype

    @property
    def state(self):
        return self.inner.state

    def _full_step(self, frame: Image.Image, t0: float, flow_max: float) -> Tuple[Image.Image, SpecTiming]:
        out_img, t = self.inner.step(frame)
        # Pull the latent from the inner state (EditWorker.step pushes it onto
        # state.prev_latents).
        last = self.inner.state.last_latent()
        if last is not None:
            self._prev_latent = last
        self._consec_hits = 0
        spec = SpecTiming(
            te_ms=t.te_ms, vae_enc_ms=t.vae_enc_ms,
            denoise_ms=t.denoise_ms, vae_dec_ms=t.vae_dec_ms,
            total_ms=(time.perf_counter() - t0) * 1000,
            accepted=False, flow_max=flow_max,
        )
        return out_img, spec

    @torch.no_grad()
    def step(self, frame: Image.Image) -> Tuple[Image.Image, SpecTiming]:
        t0 = time.perf_counter()
        curr_rgb = np.asarray(frame.convert("RGB"))
        curr_gray = flowlib.to_gray(curr_rgb)

        # First frame: nothing to speculate from -> miss.
        if self._prev_input_gray is None or self._prev_latent is None:
            self._prev_input_gray = curr_gray
            return self._full_step(frame, t0, flow_max=0.0)

        # Compute flow + decide
        flow = flowlib.farneback_flow(self._prev_input_gray, curr_gray)
        flow_mag = np.linalg.norm(flow, axis=2)
        flow_max = float(flow_mag.max())

        force_miss = (
            flow_max > self.flow_thresh or self._consec_hits >= self.max_consec
        )
        if force_miss:
            self._prev_input_gray = curr_gray
            return self._full_step(frame, t0, flow_max=flow_max)

        # ------ Speculation path ------
        spec = SpecTiming(flow_max=flow_max, accepted=True)
        ts = time.perf_counter()
        speculated = flowlib.warp_latent(
            self._prev_latent, flow, self.inner.pipeline.vae_scale_factor,
        )
        spec.denoise_ms = (time.perf_counter() - ts) * 1000  # report warp time in denoise slot
        ts = time.perf_counter()
        images = ops.decode_latent(self.inner.pipeline, speculated, output_type="pil")
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        spec.vae_dec_ms = (time.perf_counter() - ts) * 1000

        # Update state for next iteration.
        self._prev_latent = speculated
        self._prev_input_gray = curr_gray
        # Also push to inner worker's SharedState so anything reading
        # last_latent() (e.g. metric loops) sees the speculated value.
        self.inner.state.push_latent(speculated)
        self._consec_hits += 1

        spec.total_ms = (time.perf_counter() - t0) * 1000
        return images[0], spec
