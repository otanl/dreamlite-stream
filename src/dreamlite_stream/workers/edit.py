"""EditWorker — wraps DreamLiteMobilePipeline as a per-frame i2i step.

Init-noise modes:
  pure : init = randn (default)
  prev : init = sqrt(1-s) * prev_latent + sqrt(s) * randn  (s = noise_strength)

Acceleration knobs (independent, can be combined):
  compile=True    -> torch.compile(unet) for 1.5-2.5x denoise speedup
  prefetch flow   -> use prefetch() + step_with_prefetch() to overlap
                     TE_{n+1} + VAE_enc_{n+1} on a side CUDA stream with
                     denoise_n + VAE_dec_n on the default stream
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from PIL import Image

import cv2
import numpy as np

from .. import pipeline_ops as ops
from .. import flow as flowlib
from ..state import SharedState


@dataclass
class StepTiming:
    te_ms: float = 0.0
    vae_enc_ms: float = 0.0
    denoise_ms: float = 0.0
    vae_dec_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class Prefetched:
    """Tensors produced on the side stream by EditWorker.prefetch().

    `done` is a CUDA event recorded after both TE and VAE encode finish on the
    side stream. Consumers must wait_event(done) before reading the tensors.
    """

    prompt_embeds: torch.Tensor
    prompt_mask: torch.Tensor
    ref_latent: torch.Tensor
    done: Optional[torch.cuda.Event] = None
    ev_start: Optional[torch.cuda.Event] = None
    ev_after_te: Optional[torch.cuda.Event] = None


@dataclass
class EditWorker:
    pipeline: object
    state: SharedState
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    init_mode: str = "pure"  # 'pure' | 'prev'  (prev is degenerate, kept for ablation)
    noise_strength: float = 0.7
    seed: Optional[int] = None
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_backend: Optional[str] = None  # None = inductor (default); 'cudagraphs' avoids Triton

    # MVP-2 cond_image_latents = w_input*input + w_prev*warp(prev,flow) + w_kf*keyframe.
    # Defaults reproduce MVP-1 baseline (cond = current input frame).
    w_input: float = 1.0
    w_prev: float = 0.0
    w_kf: float = 0.0

    # Optional temporal LLLite controller. When set, before each denoise the
    # warped previous decoded frame is fed to controller.set_cond_image() so
    # the adapter can inject temporal-consistency δ into the UNet attention.
    lllite_controller: Optional[object] = None

    _side_stream: Optional[torch.cuda.Stream] = field(default=None, init=False, repr=False)
    _compiled: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            self._side_stream = torch.cuda.Stream()
        if self.compile:
            self._enable_compile()

    def _enable_compile(self) -> None:
        if hasattr(self.pipeline.unet, "_orig_mod"):
            self._compiled = True
            return
        kwargs = dict(fullgraph=False, dynamic=False)
        if self.compile_backend is not None:
            kwargs["backend"] = self.compile_backend
        else:
            kwargs["mode"] = self.compile_mode
        try:
            self.pipeline.unet = torch.compile(self.pipeline.unet, **kwargs)
            self._compiled = True
            print(
                f"[compile] unet wrapped ({kwargs}); first denoise will pay compile cost"
            )
        except Exception as e:
            print(
                f"[compile] WARN: torch.compile failed "
                f"({type(e).__name__}: {e}); falling back to eager"
            )

    def _ensure_timesteps(self) -> None:
        s = self.state
        timesteps, _ = ops.set_timesteps(
            self.pipeline,
            height=s.height,
            width=s.width,
            num_inference_steps=s.num_inference_steps,
            device=self.device,
        )
        s.cached_timesteps = timesteps
        if s.cached_time_ids is None:
            s.cached_time_ids = ops.make_time_ids(s.width, s.height, self.device, self.dtype)

    def _gpu_sync(self) -> None:
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def _set_lllite_cond(self, frame: Image.Image) -> None:
        """Compute warped previous decoded RGB and feed it to the LLLite
        controller as cond_image. No-op if no controller, or if this is the
        first frame (no prev decoded output yet)."""
        if self.lllite_controller is None:
            return
        s = self.state
        prev = s.prev_decoded_rgb
        curr_rgb = np.asarray(frame.convert("RGB"))
        if prev is None or s.prev_input_gray is None:
            # First frame: no warp possible — feed the current input itself
            # as a benign cond (an unwarped image close to what target_t+1
            # would look like under heavy blending).
            warped = curr_rgb
        else:
            curr_gray = flowlib.to_gray(curr_rgb)
            flow = flowlib.farneback_flow(s.prev_input_gray, curr_gray)
            H, W = flow.shape[:2]
            xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            # Forward warp prev_decoded to curr-frame coords: MINUS flow.
            warped = cv2.remap(
                prev, xs - flow[..., 0], ys - flow[..., 1],
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
        # (H, W, 3) uint8 -> (1, 3, H, W) float [-1, 1] on device
        t = torch.from_numpy(warped).permute(2, 0, 1).contiguous().float() / 255.0
        t = (t * 2.0 - 1.0).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        self.lllite_controller.set_cond_image(t)

    def _build_cond_image_latent(
        self, ref_latent: torch.Tensor, frame: Image.Image,
    ) -> torch.Tensor:
        """Blend (current input latent, flow-warped prev latent, keyframe).

        When weights are (1, 0, 0) this returns ref_latent unchanged — i.e.
        the MVP-1 baseline behaviour. Non-zero w_prev / w_kf engages the
        MVP-2 temporal mechanisms.

        Side effect: updates state.prev_input_gray for the next call's flow.
        """
        s = self.state
        wi, wp, wk = self.w_input, self.w_prev, self.w_kf
        prev = s.last_latent()
        kf = s.keyframe_latent

        # Cheap fast-path: pure baseline.
        if wp == 0.0 and wk == 0.0:
            # still update prev_input_gray so future calls can swap modes.
            s.prev_input_gray = flowlib.to_gray(np.asarray(frame))
            return ref_latent

        # Compute warped prev latent (or fall back to ref_latent at frame 0).
        curr_gray = flowlib.to_gray(np.asarray(frame))
        if wp > 0.0 and prev is not None and s.prev_input_gray is not None:
            flow = flowlib.farneback_flow(s.prev_input_gray, curr_gray)
            warped_prev = flowlib.warp_latent(prev, flow, self.pipeline.vae_scale_factor)
        elif wp > 0.0 and prev is not None:
            warped_prev = prev  # no flow available yet; use prev unwarped
        else:
            warped_prev = ref_latent  # frame 0 fallback

        # Keyframe fallback when not yet generated.
        if wk > 0.0 and kf is not None:
            kf_lat = kf
        else:
            kf_lat = ref_latent  # fallback so its weight gets folded in

        total = wi + wp + wk
        if total <= 0:
            cond = ref_latent
        else:
            cond = (wi * ref_latent + wp * warped_prev + wk * kf_lat) / total

        s.prev_input_gray = curr_gray
        return cond

    def _build_init_latents(self, ref_latent: torch.Tensor) -> torch.Tensor:
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(self.seed + self.state.frame_idx)
        noise = ops.make_init_noise(
            self.pipeline,
            height=self.state.height,
            width=self.state.width,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        if self.init_mode == "pure" or self.state.last_latent() is None:
            return noise
        s = max(0.0, min(1.0, self.noise_strength))
        prev = self.state.last_latent()
        return (1.0 - s) ** 0.5 * prev + (s ** 0.5) * noise

    @torch.no_grad()
    def step(self, frame: Image.Image) -> Tuple[Image.Image, StepTiming]:
        """Synchronous, non-pipelined path (baseline)."""
        self._ensure_timesteps()
        s = self.state
        timing = StepTiming()
        t0 = time.perf_counter()

        self._gpu_sync(); t = time.perf_counter()
        prompt_embeds, prompt_mask = ops.encode_prompt_edit(
            self.pipeline, s.prompt, frame, self.device, self.dtype,
        )
        self._gpu_sync(); timing.te_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        ref_latent = ops.encode_image_to_latent(
            self.pipeline, frame, s.height, s.width, self.device, self.dtype,
        )
        self._gpu_sync(); timing.vae_enc_ms = (time.perf_counter() - t) * 1000

        cond_image_latent = self._build_cond_image_latent(ref_latent, frame)
        self._set_lllite_cond(frame)
        init_latents = self._build_init_latents(ref_latent)
        t = time.perf_counter()
        out_latent = ops.denoise(
            self.pipeline, init_latents, cond_image_latent,
            prompt_embeds, prompt_mask,
            s.cached_time_ids, s.cached_timesteps,
        )
        self._gpu_sync(); timing.denoise_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        images = ops.decode_latent(self.pipeline, out_latent, output_type="pil")
        self._gpu_sync(); timing.vae_dec_ms = (time.perf_counter() - t) * 1000

        s.push_latent(out_latent)
        if self.lllite_controller is not None:
            s.prev_decoded_rgb = np.asarray(images[0].convert("RGB"))
        timing.total_ms = (time.perf_counter() - t0) * 1000
        return images[0], timing

    @torch.no_grad()
    def prefetch(self, frame: Image.Image) -> Prefetched:
        """Kick off TE + VAE_enc on a side stream. Returns immediately on cuda."""
        s = self.state
        if self._side_stream is None:
            te, mask = ops.encode_prompt_edit(
                self.pipeline, s.prompt, frame, self.device, self.dtype,
            )
            ref = ops.encode_image_to_latent(
                self.pipeline, frame, s.height, s.width, self.device, self.dtype,
            )
            return Prefetched(prompt_embeds=te, prompt_mask=mask, ref_latent=ref)

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_after_te = torch.cuda.Event(enable_timing=True)
        ev_done = torch.cuda.Event(enable_timing=True)
        side = self._side_stream
        with torch.cuda.stream(side):
            ev_start.record(side)
            te, mask = ops.encode_prompt_edit(
                self.pipeline, s.prompt, frame, self.device, self.dtype,
            )
            ev_after_te.record(side)
            ref = ops.encode_image_to_latent(
                self.pipeline, frame, s.height, s.width, self.device, self.dtype,
            )
            ev_done.record(side)
        return Prefetched(
            prompt_embeds=te, prompt_mask=mask, ref_latent=ref,
            done=ev_done, ev_start=ev_start, ev_after_te=ev_after_te,
        )

    @torch.no_grad()
    def step_with_prefetch(
        self, frame: Image.Image, prefetched: Prefetched,
    ) -> Tuple[Image.Image, StepTiming]:
        """Consume a Prefetched bundle and run denoise + decode on default stream."""
        self._ensure_timesteps()
        s = self.state
        timing = StepTiming()
        t0 = time.perf_counter()

        if prefetched.done is not None:
            torch.cuda.current_stream().wait_event(prefetched.done)

        cond_image_latent = self._build_cond_image_latent(prefetched.ref_latent, frame)
        self._set_lllite_cond(frame)
        init_latents = self._build_init_latents(prefetched.ref_latent)
        t = time.perf_counter()
        out_latent = ops.denoise(
            self.pipeline, init_latents, cond_image_latent,
            prefetched.prompt_embeds, prefetched.prompt_mask,
            s.cached_time_ids, s.cached_timesteps,
        )
        self._gpu_sync(); timing.denoise_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        images = ops.decode_latent(self.pipeline, out_latent, output_type="pil")
        self._gpu_sync(); timing.vae_dec_ms = (time.perf_counter() - t) * 1000

        # Read side-stream timing (events have completed before our sync above)
        if prefetched.ev_start is not None and prefetched.done is not None:
            timing.te_ms = prefetched.ev_start.elapsed_time(prefetched.ev_after_te)
            timing.vae_enc_ms = prefetched.ev_after_te.elapsed_time(prefetched.done)

        s.push_latent(out_latent)
        if self.lllite_controller is not None:
            s.prev_decoded_rgb = np.asarray(images[0].convert("RGB"))
        timing.total_ms = (time.perf_counter() - t0) * 1000
        return images[0], timing
