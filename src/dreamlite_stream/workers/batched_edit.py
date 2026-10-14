"""BatchedEditWorker — buffer N frames and process them as one UNet call.

Exploits asymmetric batch scalability between the heavy Qwen3-VL TE
(uncompiled, kernel-launch-bound, ~3-4x throughput at B=4) and the
compiled UNet (saturated, ~1.2x throughput at B=4). Net per-frame
throughput improves substantially over per-frame inference because the
TE bottleneck is hidden inside the batched call.

Trade-off: +(N-1) frames of latency. At 22 FPS with N=4, that's ~135 ms
buffering delay before the first output of each batch is emitted. For
non-interactive video editing this is acceptable.

An LLLite controller is optional (`lllite_controller`); its cond_emb buffers
must be sized for this worker's batch (`max_batch_size == batch_size`).

Opt-in options that trade per-frame recomputation for throughput, none of
which change the default path:

  te_refresh_every=0   encode the prompt + image once and reuse it
  cond_refresh_every=0 set the LLLite cond once and hold it
  shared_noise         one init-noise draw for every frame of every batch
  fast_vae             fp16 / channels_last VAE with GPU-side
                       (de)normalisation instead of the PIL image_processor
  vae_compile          torch.compile that VAE
  output_type="np"     return uint8 arrays instead of PIL images

`reset_sequence()` clears the per-clip state those options hold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .. import flow as flowlib
from .. import pipeline_ops as ops
from ..state import SharedState


@dataclass
class BatchTiming:
    te_ms: float = 0.0
    vae_enc_ms: float = 0.0
    denoise_ms: float = 0.0
    vae_dec_ms: float = 0.0
    total_ms: float = 0.0
    n_frames: int = 0
    # Wall time the caller was blocked inside the prefetch call that produced
    # this batch's inputs. NOT part of total_ms: in a prefetch-then-step loop
    # that call runs one iteration earlier, in front of the previous batch's
    # step, so a step-only clock never sees it.
    prefetch_ms: float = 0.0

    @property
    def per_frame_ms(self) -> float:
        return self.total_ms / max(self.n_frames, 1)

    @property
    def loop_ms(self) -> float:
        """Caller-side cost of this batch in a prefetch-then-step loop.

        Use this, not total_ms, for end-to-end throughput: frames / sum(loop_ms).
        """
        return self.total_ms + self.prefetch_ms


@dataclass
class PrefetchedBatch:
    """Tensors produced on the side stream by BatchedEditWorker.prefetch_batch.

    `done` is a CUDA event recorded after both TE and VAE encode finish on
    the side stream. Consumers must wait_event(done) before reading.
    """

    prompt_embeds: torch.Tensor    # (B, L, D)
    prompt_mask: torch.Tensor      # (B, L)
    ref_latents: torch.Tensor      # (B, C, h, w)
    n_frames: int = 0
    done: Optional[torch.cuda.Event] = None
    ev_start: Optional[torch.cuda.Event] = None
    ev_after_te: Optional[torch.cuda.Event] = None
    # Async future for the CPU-side per-frame optical flow used to build
    # the LLLite cond image. None when refresh is not due, or when the
    # worker pool is disabled. When set, step_batch_with_prefetch waits
    # on the future before calling set_cond_image so the CPU flow cost
    # overlaps with GPU TE/VAE_enc on the side stream.
    cond_rgbs_future: Optional[object] = None
    # Wall time the producing call (prefetch_batch / prefetch_gpu_kick) held
    # its caller: host-side preprocessing plus any host sync inside the TE
    # forward. Copied into BatchTiming.prefetch_ms by the consuming step.
    host_ms: float = 0.0


@dataclass
class BatchedEditWorker:
    pipeline: object
    state: SharedState
    batch_size: int = 4
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    seed: Optional[int] = None
    compile: bool = False
    compile_mode: str = "reduce-overhead"

    lllite_controller: Optional[object] = None
    # Fixed-noise mode (streamdiffusion-mac trick): use the SAME noise pattern
    # for every frame so per-frame latent variation comes only from input
    # changes. Reduces flicker without adding compute.
    fixed_noise: bool = False
    # One init-noise draw shared by EVERY frame of every batch (a single
    # (1, C, h, w) tensor expanded to B). fixed_noise above only repeats the
    # same (B, C, h, w) tensor batch after batch, so each batch slot keeps
    # its own pattern and neighbouring frames still get different noise;
    # shared_noise removes that difference too. On a static input the
    # output is then identical from frame to frame.
    shared_noise: bool = False
    # Refresh the LLLite cond_emb buffer only every N batches; reuse the
    # previous embedding in between. Hides the 108-hook CNN cost when N>1
    # without retraining (the previous cond is a good approximation when
    # consecutive batches see similar scenes). 0 = set it on the first
    # batch after construction / reset_sequence() and hold it from then on
    # (no optical flow, no cond CNN, no previous-frame tracking).
    cond_refresh_every: int = 1

    # Refresh the Q3-VL TE prompt_embeds only every N batches; reuse them
    # in between. Amortises the dominant per-batch GPU cost on the side
    # stream (~250 ms TE forward at B=8 on a 3090Ti) when the prompt is
    # static across frames. The image-conditioned part of the TE drifts
    # slowly under fixed prompts so a refresh schedule similar to
    # cond_refresh_every is safe in practice. 1 = no caching. 0 = encode
    # once and reuse until reset_sequence() or a forced refresh (prompt
    # change); the TE then leaves the per-batch path entirely.
    te_refresh_every: int = 1

    # On TE refresh, run Q3-VL on a SINGLE representative frame
    # (``frames[0]``) and broadcast the resulting embeddings to the full
    # batch via .expand+.contiguous, instead of processing all B images.
    # Vision-encoder compute drops ~B× (the multimodal sequence is much
    # shorter), shrinking the refresh GPU wall from ~250 ms to ~80–120 ms
    # at B=8. Safe when consecutive frames share scene content (live
    # camera at 30 fps); the cache is reused for `te_refresh_every`
    # batches anyway so per-frame image-aware conditioning is already
    # being amortised, this just extends that to within-batch as well.
    te_batch_one: bool = False

    # Worker pool for per-frame optical-flow computation during cond
    # refresh. Default 8 workers; set 0 to disable threading.
    cond_flow_workers: int = 8

    # Run VAE encode/decode through a private fp16 / channels_last copy of
    # the VAE, with normalisation and uint8 quantisation done on the GPU,
    # instead of pipeline.image_processor (PIL -> float on the host going
    # in, float -> PIL coming out: ~9 ms per 512x512 frame on the calling
    # thread). pipeline.vae itself is left untouched, so other workers
    # sharing the pipeline are unaffected. Frames may be PIL images or
    # uint8 (H, W, 3) arrays either way.
    fast_vae: bool = False
    # Additionally torch.compile the private VAE with autotuned kernels and
    # no CUDA graphs (so it carries no thread affinity). Compiled once at
    # construction for (batch_size, height, width). Requires fast_vae.
    vae_compile: bool = False
    # "pil" returns PIL images. "np" returns uint8 (H, W, 3) RGB arrays,
    # which saves a PIL round trip when the caller wants arrays anyway.
    output_type: str = "pil"

    _compiled: bool = field(default=False, init=False, repr=False)
    _vae_enc: Optional[object] = field(default=None, init=False, repr=False)
    _vae_dec: Optional[object] = field(default=None, init=False, repr=False)
    _shared_noise_cache: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _side_stream: Optional[torch.cuda.Stream] = field(default=None, init=False, repr=False)
    # Dedicated CUDA stream for the background Q3-VL TE forward (driven by
    # the bg cpu_prep_pool thread). Lets the .item() sync inside Q3-VL's
    # rot_pos_emb block ONLY the bg thread, while the main pipeline thread
    # keeps issuing UNet steps on the main stream.
    _te_side_stream: Optional[torch.cuda.Stream] = field(default=None, init=False, repr=False)
    _last_prev_decoded: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _last_prev_input_gray: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cond_call_idx: int = field(default=0, init=False, repr=False)
    _cond_flow_pool: Optional[object] = field(default=None, init=False, repr=False)
    _te_call_idx: int = field(default=0, init=False, repr=False)
    _cached_te_embeds: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _cached_te_mask: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    # Set externally (e.g. by the demo's prompt-change key handler) to
    # force the next prefetch_cpu_only to refresh the TE cache regardless
    # of the periodic schedule. The flag is cleared inside the refresh.
    _force_te_refresh: bool = field(default=False, init=False, repr=False)
    # CUDA event recorded on whichever stream most recently produced the
    # cached TE tensors. ``prefetch_gpu_kick`` waits on this before cloning
    # the cache so the main side stream doesn't read partial TE-forward
    # results from the bg thread's te_side_stream. None until the first
    # refresh; harmless no-op when the producer stream == consumer stream.
    _last_cache_te_ev: Optional[torch.cuda.Event] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.output_type not in ("pil", "np"):
            raise ValueError(f"output_type must be 'pil' or 'np', got {self.output_type!r}")
        if self.vae_compile and not self.fast_vae:
            raise ValueError("vae_compile requires fast_vae")
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            self._side_stream = torch.cuda.Stream()
            self._te_side_stream = torch.cuda.Stream()
        if self.compile:
            self._enable_compile()
        if self.fast_vae:
            self._enable_fast_vae()
        if self.cond_flow_workers and self.cond_flow_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            self._cond_flow_pool = ThreadPoolExecutor(max_workers=self.cond_flow_workers)

    def reset_sequence(self) -> None:
        """Forget everything tied to the clip just processed.

        Call between unrelated clips when one worker is reused. Restarts
        the frame counter (and with it the per-batch noise seeds), drops
        the LLLite previous-frame state and restarts its refresh schedule
        so the next clip sets its own cond, and drops the TE cache so the
        next prefetch re-encodes. The compiled UNet, the VAE and the shared
        noise draw are untouched.
        """
        self.state.frame_idx = 0
        self._last_prev_decoded = None
        self._last_prev_input_gray = None
        self._cond_call_idx = 0
        self._te_call_idx = 0
        self._cached_te_embeds = None
        self._cached_te_mask = None
        self._last_cache_te_ev = None
        self._force_te_refresh = False

    def _te_refresh_due(self) -> bool:
        """Consume one TE tick; True if this call must recompute the cache.

        Shared by prefetch_batch and prefetch_cpu_only so both follow the
        same schedule. te_refresh_every=0 never refreshes on schedule: only
        the first call and a forced refresh do.
        """
        due = (
            self._cached_te_embeds is None
            or self._force_te_refresh
            or (self.te_refresh_every > 0 and self._te_call_idx % self.te_refresh_every == 0)
        )
        self._te_call_idx += 1
        self._force_te_refresh = False
        return due

    def _cond_refresh_due(self) -> bool:
        """True if the NEXT set_cond_image call rebuilds the LLLite cond.

        Does not advance the counter; _set_lllite_cond_for_batch does.
        cond_refresh_every=0 rebuilds only on the first call.
        """
        return self._cond_call_idx == 0 or (
            self.cond_refresh_every > 0
            and self._cond_call_idx % self.cond_refresh_every == 0
        )

    def _maybe_kick_cond_flow(self, frames: List[Image.Image]):
        """If LLLite is active AND refresh is due on the NEXT cond-set call,
        kick the per-frame optical-flow computation in a worker thread now.
        Returns a Future yielding a list of warped uint8 RGB arrays, or None
        when no refresh is needed (the next set_cond_image will reuse the
        cached emb).
        """
        if self.lllite_controller is None or self._cond_flow_pool is None:
            return None
        # The next set_cond_image call (inside step) will refresh iff
        # _cond_refresh_due(). We read the SAME index here; the increment
        # happens inside _set_lllite_cond_for_batch.
        if not self._cond_refresh_due():
            return None
        if self._last_prev_decoded is None or self._last_prev_input_gray is None:
            # First batch: trivial cond (input frames themselves).
            def _identity():
                return [ops.as_rgb_array(f) for f in frames]
            return self._cond_flow_pool.submit(_identity)
        prev_rgb = self._last_prev_decoded
        prev_gray = self._last_prev_input_gray

        def _flow_one(f):
            curr_rgb = ops.as_rgb_array(f)
            curr_gray = flowlib.to_gray(curr_rgb)
            flow = flowlib.farneback_flow(prev_gray, curr_gray)
            H, W = flow.shape[:2]
            xs, ys = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
            return cv2.remap(
                prev_rgb, xs - flow[..., 0], ys - flow[..., 1],
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )

        # Submit each frame as a separate task; gather results in order.
        # Returns a Future that resolves to the list of warped uint8 RGBs.
        futures = [self._cond_flow_pool.submit(_flow_one, f) for f in frames]
        class _Joined:
            def result(self):
                return [fut.result() for fut in futures]
        return _Joined()

    def _set_lllite_cond_for_batch(
        self, frames: List[Image.Image], precomputed_cond_rgbs=None,
    ) -> None:
        """Build per-frame cond images for batched LLLite.

        If `precomputed_cond_rgbs` is provided (from a future kicked by
        prefetch_batch), use it directly — the CPU flow has already run in
        parallel with GPU TE/VAE_enc on the side stream. Otherwise compute
        inline (threaded if a worker pool is available).
        """
        if self.lllite_controller is None:
            return
        due = self._cond_refresh_due()
        self._cond_call_idx += 1
        if not due:
            return
        if precomputed_cond_rgbs is not None:
            cond_rgbs = precomputed_cond_rgbs
        elif self._last_prev_decoded is None or self._last_prev_input_gray is None:
            cond_rgbs = [ops.as_rgb_array(f) for f in frames]
        else:
            prev_rgb = self._last_prev_decoded
            prev_gray = self._last_prev_input_gray

            def _flow_one(f):
                curr_rgb = ops.as_rgb_array(f)
                curr_gray = flowlib.to_gray(curr_rgb)
                flow = flowlib.farneback_flow(prev_gray, curr_gray)
                H, W = flow.shape[:2]
                xs, ys = np.meshgrid(np.arange(W, dtype=np.float32),
                                     np.arange(H, dtype=np.float32))
                return cv2.remap(
                    prev_rgb, xs - flow[..., 0], ys - flow[..., 1],
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                )

            if self._cond_flow_pool is not None:
                cond_rgbs = list(self._cond_flow_pool.map(_flow_one, frames))
            else:
                cond_rgbs = [_flow_one(f) for f in frames]
        stack = np.stack(cond_rgbs, axis=0).astype(np.float32) / 255.0
        stack = stack * 2.0 - 1.0
        t = torch.from_numpy(stack).permute(0, 3, 1, 2).contiguous()
        t = t.to(device=self.device, dtype=self.dtype)
        self.lllite_controller.set_cond_image(t)

    def _track_decoded_for_lllite(self, frames: List[Image.Image], outputs: List[Image.Image]) -> None:
        """Save the LAST frame's decoded RGB and input gray for the next batch."""
        if self.lllite_controller is None:
            return
        if self.cond_refresh_every == 0:
            # The cond is held after its first set, so nothing will read these.
            return
        last_input = ops.as_rgb_array(frames[-1])
        last_output = ops.as_rgb_array(outputs[-1])
        self._last_prev_decoded = last_output
        self._last_prev_input_gray = flowlib.to_gray(last_input)

    def _enable_fast_vae(self) -> None:
        """Build the private fp16 / channels_last VAE, optionally compiled."""
        import copy

        vae = copy.deepcopy(self.pipeline.vae)
        vae = vae.to(device=self.device, dtype=torch.float16)
        vae = vae.to(memory_format=torch.channels_last).eval()
        self._vae_enc, self._vae_dec = vae.encoder, vae.decoder
        if not self.vae_compile:
            return
        # Autotuning is what makes compiling this VAE worthwhile: for some of
        # its convs Inductor's Triton kernels beat cuDNN by a wide margin.
        # It prints a results table per conv; keep that out of the console.
        sa = None
        try:
            import torch._inductor.select_algorithm as sa
        except Exception:
            pass
        prev_print = getattr(sa, "PRINT_AUTOTUNE", None)
        try:
            if prev_print is not None:
                sa.PRINT_AUTOTUNE = False
            t0 = time.perf_counter()
            enc = torch.compile(vae.encoder, mode="max-autotune-no-cudagraphs")
            dec = torch.compile(vae.decoder, mode="max-autotune-no-cudagraphs")
            s = self.state
            x = torch.zeros(
                self.batch_size, 3, s.height, s.width,
                device=self.device, dtype=torch.float16,
            ).contiguous(memory_format=torch.channels_last)
            with torch.no_grad():
                for _ in range(2):
                    dec(enc(x))
            self._vae_enc, self._vae_dec = enc, dec
            print(
                f"[compile] vae encoder+decoder compiled for "
                f"({self.batch_size}, {s.height}, {s.width}) in "
                f"{time.perf_counter() - t0:.0f}s"
            )
        except Exception as e:
            print(f"[compile] WARN: VAE compile failed ({e}); eager fp16 VAE")
        finally:
            if prev_print is not None:
                sa.PRINT_AUTOTUNE = prev_print

    def _encode_ref_latents(self, frames: List[Image.Image]) -> torch.Tensor:
        """VAE-encode this batch's frames into reference latents."""
        s = self.state
        if self._vae_enc is not None:
            frames_u8 = ops.frames_to_uint8_batch(frames, s.height, s.width)
            return ops.encode_uint8_batch_to_latent(
                self._vae_enc, frames_u8, self.device, torch.float16, self.dtype,
            )
        return ops.encode_image_to_latent_batch(
            self.pipeline, frames, s.height, s.width, self.device, self.dtype,
        )

    def _encode_prepared(self, vae_input: torch.Tensor) -> torch.Tensor:
        """VAE-encode the host tensor prefetch_cpu_only prepared for this batch.

        That tensor is a uint8 (B, H, W, 3) stack under fast_vae and the
        image_processor's normalised float tensor otherwise.
        """
        if self._vae_enc is not None:
            return ops.encode_uint8_batch_to_latent(
                self._vae_enc, vae_input, self.device, torch.float16, self.dtype,
            )
        vae_input = vae_input.to(device=self.device, dtype=self.dtype)
        return ops.retrieve_latents(
            self.pipeline.vae.encode(vae_input), sample_mode="argmax",
        )

    def _decode(self, latents: torch.Tensor) -> list:
        """Decode latents to output frames in `output_type`."""
        if self._vae_dec is not None:
            arr = ops.decode_latent_to_uint8(
                self.pipeline, self._vae_dec, latents, torch.float16,
            )
            if self.output_type == "np":
                return list(arr)
            return [Image.fromarray(a) for a in arr]
        images = ops.decode_latent(self.pipeline, latents, output_type="pil")
        if self.output_type == "np":
            return [np.asarray(im) for im in images]
        return images

    def _make_init_latents(self, B: int) -> torch.Tensor:
        """Init noise for a batch of B frames under the configured policy."""
        s = self.state
        if self.shared_noise:
            cached = self._shared_noise_cache
            vae_scale = self.pipeline.vae_scale_factor
            lat_hw = (s.height // vae_scale, s.width // vae_scale)
            if cached is None or tuple(cached.shape[-2:]) != lat_hw:
                gen = None
                if self.seed is not None:
                    gen = torch.Generator(device="cpu").manual_seed(self.seed)
                cached = ops.make_init_noise(
                    self.pipeline, height=s.height, width=s.width,
                    device=self.device, dtype=self.dtype, generator=gen, batch_size=1,
                )
                self._shared_noise_cache = cached
            # .contiguous() materialises a fresh B-major tensor per call, so
            # the compiled UNet never sees a stride-0 broadcast view.
            return cached.expand(B, -1, -1, -1).contiguous()
        gen = None
        if self.seed is not None:
            seed_offset = 0 if self.fixed_noise else s.frame_idx
            gen = torch.Generator(device="cpu").manual_seed(self.seed + seed_offset)
        return ops.make_init_noise(
            self.pipeline, height=s.height, width=s.width,
            device=self.device, dtype=self.dtype, generator=gen, batch_size=B,
        )

    def _enable_compile(self) -> None:
        # Skip if already compiled by a previous worker — re-wrapping would
        # chain OptimizedModule on OptimizedModule, breaking attribute
        # delegation (.config etc.) and re-paying the compile cost.
        if getattr(self.pipeline.unet, "is_trt_engine", False):
            return  # a TensorRT engine stands in for the UNet; nothing to compile
        if hasattr(self.pipeline.unet, "_orig_mod"):
            self._compiled = True
            return
        try:
            self.pipeline.unet = torch.compile(
                self.pipeline.unet,
                mode=self.compile_mode, fullgraph=False, dynamic=False,
            )
            self._compiled = True
            print(
                f"[compile] unet wrapped (mode={self.compile_mode}, "
                f"batch_size={self.batch_size}); first batch pays compile cost"
            )
        except Exception as e:
            print(f"[compile] WARN: torch.compile failed ({e}); eager fallback")

    def _gpu_sync(self) -> None:
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def _ensure_timesteps(self) -> torch.Tensor:
        s = self.state
        timesteps, _ = ops.set_timesteps(
            self.pipeline,
            height=s.height, width=s.width,
            num_inference_steps=s.num_inference_steps,
            device=self.device,
        )
        s.cached_timesteps = timesteps
        return timesteps

    @torch.no_grad()
    def step_batch(self, frames: List[Image.Image]) -> Tuple[List[Image.Image], BatchTiming]:
        s = self.state
        B = len(frames)
        prompts = [s.prompt] * B
        timing = BatchTiming(n_frames=B)
        t0 = time.perf_counter()

        timesteps = self._ensure_timesteps()
        time_ids = ops.make_time_ids(s.width, s.height, self.device, self.dtype).expand(B, -1)

        # TE (the headline win)
        self._gpu_sync(); t = time.perf_counter()
        prompt_embeds, prompt_mask = ops.encode_prompt_edit_batch(
            self.pipeline, prompts, frames, self.device, self.dtype,
        )
        self._gpu_sync(); timing.te_ms = (time.perf_counter() - t) * 1000

        # VAE encode (also batched, sub-linear like TE)
        t = time.perf_counter()
        ref_latents = self._encode_ref_latents(frames)
        self._gpu_sync(); timing.vae_enc_ms = (time.perf_counter() - t) * 1000

        # Init noise (batched)
        init_latents = self._make_init_latents(B)

        # Feed per-frame cond images to LLLite (no-op if no controller)
        self._set_lllite_cond_for_batch(frames)

        # Denoise (existing op handles batch>1)
        t = time.perf_counter()
        out_latents = ops.denoise(
            self.pipeline,
            init_latents=init_latents,
            cond_image_latents=ref_latents,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            time_ids=time_ids,
            timesteps=timesteps,
        )
        self._gpu_sync(); timing.denoise_ms = (time.perf_counter() - t) * 1000

        # Decode (returns list)
        t = time.perf_counter()
        images = self._decode(out_latents)
        self._gpu_sync(); timing.vae_dec_ms = (time.perf_counter() - t) * 1000

        self._track_decoded_for_lllite(frames, images)
        s.frame_idx += B
        timing.total_ms = (time.perf_counter() - t0) * 1000
        return images, timing

    @torch.no_grad()
    def prefetch_batch(self, frames: List[Image.Image]) -> PrefetchedBatch:
        """Kick off TE + VAE_enc for `frames` on a side CUDA stream.

        Returns immediately on cuda; the consuming default-stream call must
        `torch.cuda.current_stream().wait_event(prefetched.done)` before
        reading the returned tensors.
        """
        t_host = time.perf_counter()
        s = self.state
        B = len(frames)
        prompts = [s.prompt] * B

        # TE refresh schedule: recompute TE prompt_embeds only every
        # `te_refresh_every` batches. Cached embeds from a previous call
        # are reused in between; they're frozen GPU tensors so they don't
        # depend on the new side-stream events.
        should_refresh_te = self._te_refresh_due()

        if self._side_stream is None:
            if should_refresh_te:
                # te_batch_one: run Q3-VL on a single representative frame
                # and broadcast. Must match prefetch_cpu_only's branch so
                # the captured CUDA graph sees identical te shapes/layouts.
                if self.te_batch_one:
                    te, mask = ops.encode_prompt_edit_batch(
                        self.pipeline, [s.prompt], [frames[0]],
                        self.device, self.dtype,
                    )
                    if B > 1:
                        te = te.expand(B, -1, -1)
                        mask = mask.expand(B, -1)
                else:
                    te, mask = ops.encode_prompt_edit_batch(
                        self.pipeline, prompts, frames, self.device, self.dtype,
                    )
                # Force contiguous so the cached tensor and any subsequent
                # .clone() share the same layout. Without this the sliced
                # tensor (storage_offset != 0) and its clone (contiguous,
                # offset 0) trigger different torch.compile specializations
                # on iter 0 — which then tries to cudagraphify from the
                # pipeline thread and asserts on missing TLS state.
                te = te.contiguous()
                mask = mask.contiguous()
                self._cached_te_embeds = te
                self._cached_te_mask = mask
                # Sync path: no separate stream, so no event needed.
                self._last_cache_te_ev = None
            else:
                te = self._cached_te_embeds.clone()
                mask = self._cached_te_mask.clone()
            ref = self._encode_ref_latents(frames)
            return PrefetchedBatch(
                prompt_embeds=te, prompt_mask=mask, ref_latents=ref, n_frames=B,
                host_ms=(time.perf_counter() - t_host) * 1000,
            )

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_after_te = torch.cuda.Event(enable_timing=True)
        ev_done = torch.cuda.Event(enable_timing=True)
        side = self._side_stream

        # Kick CPU-side LLLite cond build (optical flow + warp) in parallel
        # with the GPU side-stream work, when refresh is due. The result is
        # awaited inside step_batch_with_prefetch before set_cond_image.
        cond_rgbs_future = self._maybe_kick_cond_flow(frames)

        with torch.cuda.stream(side):
            ev_start.record(side)
            if should_refresh_te:
                # te_batch_one: see note in the no-side-stream branch.
                if self.te_batch_one:
                    te, mask = ops.encode_prompt_edit_batch(
                        self.pipeline, [s.prompt], [frames[0]],
                        self.device, self.dtype,
                    )
                    if B > 1:
                        te = te.expand(B, -1, -1)
                        mask = mask.expand(B, -1)
                else:
                    te, mask = ops.encode_prompt_edit_batch(
                        self.pipeline, prompts, frames, self.device, self.dtype,
                    )
                # Force contiguous; see note in the no-side-stream branch.
                te = te.contiguous()
                mask = mask.contiguous()
                self._cached_te_embeds = te
                self._cached_te_mask = mask
            else:
                # Wait for the most recent refresh (possibly produced by the
                # bg thread on _te_side_stream) before reading the cache.
                if self._last_cache_te_ev is not None:
                    side.wait_event(self._last_cache_te_ev)
                te = self._cached_te_embeds.clone()
                mask = self._cached_te_mask.clone()
            ev_after_te.record(side)
            ref = self._encode_ref_latents(frames)
            ev_done.record(side)
        if should_refresh_te:
            # Publish ev_after_te (recorded after TE forward on _side_stream)
            # as the cache's producer event for subsequent cache-hit reads.
            self._last_cache_te_ev = ev_after_te
        return PrefetchedBatch(
            prompt_embeds=te, prompt_mask=mask, ref_latents=ref, n_frames=B,
            done=ev_done, ev_start=ev_start, ev_after_te=ev_after_te,
            cond_rgbs_future=cond_rgbs_future,
            host_ms=(time.perf_counter() - t_host) * 1000,
        )

    # ------------------------------------------------------------------
    # Split prefetch (CPU prep / GPU kick) — used by the live demo to run
    # CPU prep on a background thread concurrently with the previous batch's
    # main-stream step, so per-iter wall becomes max(prep, step) instead of
    # sum. The single-threaded `prefetch_batch` path is unchanged; these
    # two methods together produce an equivalent PrefetchedBatch.
    # ------------------------------------------------------------------

    _PROMPT_TEMPLATE = (
        "<|im_start|>system\nDescribe the key features of the input image "
        "(color, shape, size, texture, objects, background), then explain "
        "how the user's text instruction should alter or modify the image. "
        "Generate a new image that meets the user's requirements while "
        "maintaining consistency with the original input where appropriate."
        "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|>"
        "<|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
    )
    _PROMPT_DECORATION = (
        "[Edit]: A diptych with two side-by-side images of the same scene. "
        "Compared to the right side, the left one has {}"
    )
    _PROMPT_DROP_IDX = 64
    # Fixed max length for the Q3-VL processor's input_ids padding. Keeps
    # the encoder_hidden_states' sequence length constant across prompts,
    # so the captured UNet CUDA graph doesn't need re-recording on prompt
    # change. Must be large enough to fit the expanded image_pad tokens
    # (~64 for a 256×256 image) + the ~70-token system/user template
    # prefix + the actual prompt — 256 leaves comfortable room for
    # prompts up to ~120 tokens. Anything longer gets truncated.
    _PROMPT_MAX_LEN = 256

    @torch.no_grad()
    def prefetch_cpu_only(self, frames: List[Image.Image]) -> dict:
        """CPU prep + (on refresh) the full Q3-VL TE forward.

        Always runs on the bg cpu_prep_pool thread. The thread does:

        1. CPU image_processor preprocess of VAE input (always).
        2. On a refresh boundary (driven by ``te_refresh_every``):
           a. CPU tokenize + image resize for Q3-VL.
           b. Q3-VL forward on the dedicated ``_te_side_stream``. The
              intrinsic ``.item()`` sync inside ``rot_pos_emb`` blocks
              THIS bg thread, not the main pipeline thread.
           c. Slice + .contiguous() the prompt embeds, install them as
              ``_cached_te_embeds``, and record ``_last_cache_te_ev`` on
              te_side_stream so the main side stream's later clone op
              can wait on it.

        Because the main pipeline thread submitted this batch's cpu_prep
        ~1 main iter ago (~280 ms), the bg thread's TE forward (~250 ms)
        runs in parallel with the main thread's previous UNet step. By
        the time main thread waits on this future, both have finished.
        Net effect: the 270 ms refresh hitch the main thread used to pay
        every 8 batches is hidden behind a step it was already running.
        """
        s = self.state

        should_refresh_te = self._te_refresh_due()

        # VAE preprocess is always required (per-batch reference latents
        # depend on the new frames regardless of TE refresh schedule).
        if self._vae_enc is not None:
            # fast_vae: only stack to uint8 here; prefetch_gpu_kick
            # normalises on the GPU.
            vae_input_cpu = ops.frames_to_uint8_batch(frames, s.height, s.width)
        else:
            pil_frames = [ops.as_pil(img) for img in frames]
            needs_resize = any(img.size != (s.width, s.height) for img in pil_frames)
            if needs_resize:
                vae_imgs = [
                    img.resize((s.width, s.height), Image.Resampling.BILINEAR)
                    for img in pil_frames
                ]
            else:
                vae_imgs = pil_frames
            vae_input_cpu = self.pipeline.image_processor.preprocess(vae_imgs)

        if should_refresh_te:
            B = len(frames)
            # te_batch_one: run Q3-VL on just frames[0] and broadcast.
            # Multimodal seq length shrinks B×, so vision-encoder compute
            # drops correspondingly.
            if self.te_batch_one:
                te_prompts = [s.prompt]
                te_frames = [frames[0]]
            else:
                te_prompts = [s.prompt] * B
                te_frames = frames
            decorated = [self._PROMPT_DECORATION.format(p) for p in te_prompts]
            txts = [self._PROMPT_TEMPLATE.format(p) for p in decorated]
            pil_imgs = [
                ops.as_pil(img).resize((256, 256), Image.Resampling.BILINEAR)
                for img in te_frames
            ]
            # padding="max_length" fixes input_ids length to PROMPT_MAX_LEN
            # regardless of prompt content, so hidden_states (and the
            # downstream prompt_embeds shape fed to UNet) is constant
            # across prompts. Without this, a longer/shorter typed prompt
            # changes encoder_hidden_states' L dim, which forces torch.
            # compile to re-record the captured CUDA graph on the pipeline
            # thread — and that re-record can hang inside cuBLAS
            # workspace allocation during graph capture.
            tk_out_cpu = self.pipeline.processor(
                text=txts, images=pil_imgs,
                padding="max_length",
                max_length=self._PROMPT_MAX_LEN,
                truncation=True,
                return_tensors="pt",
            )

            te_side = self._te_side_stream
            drop_idx = self._PROMPT_DROP_IDX
            ev_te = torch.cuda.Event(enable_timing=False)
            with torch.cuda.stream(te_side):
                tk_out = tk_out_cpu.to(self.device)
                hidden_states = ops.prompt_hidden_state(
                    self.pipeline.text_encoder,
                    input_ids=tk_out.input_ids,
                    attention_mask=tk_out.attention_mask,
                    pixel_values=tk_out.pixel_values,
                    image_grid_thw=tk_out.image_grid_thw,
                )
                te_raw = hidden_states[:, drop_idx:].to(dtype=self.dtype)
                mask_raw = tk_out.attention_mask[:, drop_idx:].to(dtype=torch.long)
                # When te_batch_one and B>1, broadcast (1,L,D) → (B,L,D).
                # .expand is a stride-0 view; .contiguous() materialises the
                # B-batched tensor (UNet's CUDA-graph expects a real B-major
                # layout, not a broadcast view).
                if self.te_batch_one and B > 1:
                    te_raw = te_raw.expand(B, -1, -1)
                    mask_raw = mask_raw.expand(B, -1)
                te = te_raw.contiguous()
                mask = mask_raw.contiguous()
                self._cached_te_embeds = te
                self._cached_te_mask = mask
                ev_te.record(te_side)
            # Publish the event AFTER the cache assignments so any
            # subsequent reader that sees the new event also sees the
            # new tensor handles (single-writer here, but be explicit).
            self._last_cache_te_ev = ev_te

        return {
            "vae_input": vae_input_cpu,
            "n_frames": len(frames),
        }

    @torch.no_grad()
    def prefetch_gpu_kick(
        self,
        cpu_prep: dict,
        frames: Optional[List[Image.Image]] = None,
    ) -> PrefetchedBatch:
        """GPU portion of prefetch on the main pipeline thread.

        ``prefetch_cpu_only`` (run on the bg thread) has already produced
        the cached TE tensors on ``_te_side_stream`` when this iter is on
        a refresh boundary; otherwise the existing cache from a prior
        refresh is still valid. Either way, this method only needs to:

        1. wait for any in-flight TE forward (``_last_cache_te_ev``) so
           the clone below reads consistent memory across streams;
        2. clone the cached TE/mask into ``_side_stream``'s allocator so
           CUDAGraphTree gets a fresh input pointer per call (it broke
           when we passed the same external tensor repeatedly);
        3. dispatch VAE-encode on ``_side_stream`` for this batch's frames.

        Must be called from the same thread that runs
        ``step_batch_with_prefetch`` so all CUDA-graph capture/replay
        happens from one thread.

        ``host_ms`` on the result covers this call only; the CPU prep and
        TE forward ran on the bg thread and did not hold the caller.
        """
        t_host = time.perf_counter()
        B = cpu_prep["n_frames"]

        if self._side_stream is None:
            # No side stream available — synchronous fallback. The bg
            # thread already published the cache; just clone + VAE here.
            if self._last_cache_te_ev is not None:
                torch.cuda.current_stream().wait_event(self._last_cache_te_ev)
            te = self._cached_te_embeds.clone()
            mask = self._cached_te_mask.clone()
            ref = self._encode_prepared(cpu_prep["vae_input"])
            return PrefetchedBatch(
                prompt_embeds=te, prompt_mask=mask, ref_latents=ref, n_frames=B,
                host_ms=(time.perf_counter() - t_host) * 1000,
            )

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_after_te = torch.cuda.Event(enable_timing=True)
        ev_done = torch.cuda.Event(enable_timing=True)
        side = self._side_stream

        # cond-refresh path (Farneback + LLLite cond CNN) still uses the
        # original frame list, since CPU prep doesn't carry image data
        # after preprocessing.
        cond_rgbs_future = None
        if frames is not None:
            cond_rgbs_future = self._maybe_kick_cond_flow(frames)

        with torch.cuda.stream(side):
            ev_start.record(side)
            # Cross-stream sync: bg thread may have just produced the
            # cached TE on _te_side_stream; ensure side stream sees it
            # before cloning. No-op when the producer stream == side
            # (priming case via prefetch_batch).
            if self._last_cache_te_ev is not None:
                side.wait_event(self._last_cache_te_ev)
            te = self._cached_te_embeds.clone()
            mask = self._cached_te_mask.clone()
            ev_after_te.record(side)
            ref = self._encode_prepared(cpu_prep["vae_input"])
            ev_done.record(side)

        return PrefetchedBatch(
            prompt_embeds=te, prompt_mask=mask, ref_latents=ref, n_frames=B,
            done=ev_done, ev_start=ev_start, ev_after_te=ev_after_te,
            cond_rgbs_future=cond_rgbs_future,
            host_ms=(time.perf_counter() - t_host) * 1000,
        )

    @torch.no_grad()
    def step_batch_with_prefetch(
        self, frames: List[Image.Image], prefetched: PrefetchedBatch,
    ) -> Tuple[List[Image.Image], BatchTiming]:
        """Consume a PrefetchedBatch and run denoise + VAE_dec on the default
        stream. The previous batch's output is returned; meanwhile, the
        runtime should kick the NEXT batch's prefetch on the side stream so
        it overlaps with this denoise."""
        s = self.state
        B = len(frames)
        timing = BatchTiming(n_frames=B, prefetch_ms=prefetched.host_ms)
        t0 = time.perf_counter()

        if prefetched.done is not None:
            torch.cuda.current_stream().wait_event(prefetched.done)

        timesteps = self._ensure_timesteps()
        time_ids = ops.make_time_ids(s.width, s.height, self.device, self.dtype).expand(B, -1)

        init_latents = self._make_init_latents(B)

        # Feed per-frame cond images to LLLite. If prefetch kicked the CPU
        # flow async (refresh batch), await it here — the wait is hidden by
        # the GPU TE/VAE_enc on the side stream having run concurrently.
        precomputed = (
            prefetched.cond_rgbs_future.result()
            if prefetched.cond_rgbs_future is not None else None
        )
        self._set_lllite_cond_for_batch(frames, precomputed_cond_rgbs=precomputed)

        t = time.perf_counter()
        out_latents = ops.denoise(
            self.pipeline,
            init_latents=init_latents,
            cond_image_latents=prefetched.ref_latents,
            prompt_embeds=prefetched.prompt_embeds,
            prompt_mask=prefetched.prompt_mask,
            time_ids=time_ids,
            timesteps=timesteps,
        )
        self._gpu_sync(); timing.denoise_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        images = self._decode(out_latents)
        self._gpu_sync(); timing.vae_dec_ms = (time.perf_counter() - t) * 1000

        # Read side-stream timing after sync
        if prefetched.ev_start is not None and prefetched.done is not None:
            timing.te_ms = prefetched.ev_start.elapsed_time(prefetched.ev_after_te)
            timing.vae_enc_ms = prefetched.ev_after_te.elapsed_time(prefetched.done)

        self._track_decoded_for_lllite(frames, images)
        s.frame_idx += B
        timing.total_ms = (time.perf_counter() - t0) * 1000
        return images, timing
