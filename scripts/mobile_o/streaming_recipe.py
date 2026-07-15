"""Streaming recipe components R1 + R2 wired onto Mobile-O 0.5B.

task #154 (W2). Maps the dreamlite-stream recipe onto the
Mobile-O substrate:

- R1 (periodic conditioning refresh): the expensive per-frame work is
  the TE stage -- MobileCLIP vision tower -> Qwen2 LLM forward
  (output_hidden_states) -> MCP diffusion_connector. Its output
  ``encoder_hidden_states`` is constant across the whole denoise loop,
  so it is the natural cache point (exactly analogous to the LLLite
  cond cache on DreamLite-mobile). We recompute it every
  ``refresh_every`` frames and reuse it in between.

- R2 (asymmetric main/side-stream): the TE stage runs on a dedicated
  CUDA side stream while the SANA denoise loop for the *previous*
  frame occupies the main stream. Functional verification here checks
  correctness (event-synchronised handoff produces identical cond);
  steady-state overlap gains are measured later in the benchmark
  phase, not here.

Functional test (NOT a benchmark -- shared GPU, informal timings):

    .venv-mobileo/Scripts/python.exe streaming_recipe.py \
        --frames 12 --refresh-every 4 --steps 4

Checks:
  1. Cached-cond frames produce bitwise-identical cond to the frame
     the cache was filled on (trivially true; asserted as a guard).
  2. At each refresh tick, recomputed cond differs from a stale cond
     when the input frame changed (cache is actually refreshing).
  3. R2 side-stream cond equals the sequential-stream cond for the
     same input (stream handoff is race-free).
  4. End-to-end frames decode without error at reduced step count.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mobileo.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: E402
from mobileo.mm_utils import tokenizer_image_token, process_images  # noqa: E402
from mobileo.conversation import conv_templates  # noqa: E402
from mobileo.model.builder import load_pretrained_model  # noqa: E402
from diffusers.utils.torch_utils import randn_tensor  # noqa: E402


# ---------------------------------------------------------------------------
# TE stage (extracted from mobileoForInferenceLM.generate_image lines 86-117
# and the head of sample_images lines 133-136)
# ---------------------------------------------------------------------------


@torch.no_grad()
def te_forward(model, input_ids, pixel_values, with_cfg=True):
    """Vision tower + LLM forward + MCP -> encoder_hidden_states.

    Returns the float() cond tensor consumed by every denoise step.
    Mirrors generate_image()'s TE stage exactly, including the CFG
    zero-duplication that sample_images applies to the hidden tuple.
    """
    if pixel_values is not None:
        (input_ids, _, attention_mask, _, inputs_embeds, _, _) = (
            model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, und_images=pixel_values))
    else:
        attention_mask = None
        inputs_embeds = model.get_model().embed_tokens(input_ids)

    outputs = model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states
    if with_cfg:
        hidden = tuple(
            torch.cat([torch.zeros_like(h), h], dim=0) for h in hidden)
    return model.model.diffusion_connector(hidden).float()


# ---------------------------------------------------------------------------
# Denoise stage (sample_images minus the TE head, with step-count control)
# ---------------------------------------------------------------------------


@torch.no_grad()
def denoise_with_cond(model, encoder_hidden_states, with_cfg=True,
                      guidance_scale=1.5, num_inference_steps=4,
                      generator=None):
    """SANA denoise loop + VAE decode, taking a precomputed cond."""
    device = encoder_hidden_states.device
    batch_size = (encoder_hidden_states.shape[0] // 2 if with_cfg
                  else encoder_hidden_states.shape[0])

    dit = model.get_model().dit
    latents = randn_tensor(
        shape=(batch_size, dit.config.in_channels,
               dit.config.sample_size, dit.config.sample_size),
        generator=generator, device=device, dtype=torch.float32)

    model.model.noise_scheduler.set_timesteps(num_inference_steps)
    for t in model.model.noise_scheduler.timesteps:
        latent_in = torch.cat([latents] * 2) if with_cfg else latents
        if hasattr(model.model.noise_scheduler, "scale_model_input"):
            latent_in = model.model.noise_scheduler.scale_model_input(
                latent_in, t)
        noise_pred = dit(
            hidden_states=latent_in.to(torch.bfloat16),
            encoder_hidden_states=encoder_hidden_states.to(torch.bfloat16),
            timestep=t.unsqueeze(0).expand(latent_in.shape[0]).to(device),
            encoder_attention_mask=None,
        ).sample.float()
        if with_cfg:
            uncond, text = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (text - uncond)
        latents = model.model.noise_scheduler.step(
            noise_pred, t, latents).prev_sample
    return model.decode_latents(latents.to(model.model.vae.dtype))


# ---------------------------------------------------------------------------
# R1: periodic conditioning refresh
# ---------------------------------------------------------------------------


class CondRefreshCache:
    """R1: recompute the TE cond every ``refresh_every`` frames."""

    def __init__(self, model, refresh_every=4, with_cfg=True):
        self.model = model
        self.refresh_every = refresh_every
        self.with_cfg = with_cfg
        self._cond = None
        self._frame_idx = 0
        self.n_te_calls = 0

    def get(self, input_ids, pixel_values):
        if self._cond is None or self._frame_idx % self.refresh_every == 0:
            self._cond = te_forward(self.model, input_ids, pixel_values,
                                    self.with_cfg)
            self.n_te_calls += 1
        self._frame_idx += 1
        return self._cond

    def reset(self):
        self._cond = None
        self._frame_idx = 0


# ---------------------------------------------------------------------------
# R2: side-stream TE forward
# ---------------------------------------------------------------------------


class SideStreamTE:
    """R2: run te_forward on a dedicated CUDA stream.

    ``submit`` launches the TE forward asynchronously on the side
    stream; ``collect`` event-syncs and returns the cond. The caller
    overlaps `submit(frame k+1)` with the main-stream denoise of
    frame k.
    """

    def __init__(self, model, with_cfg=True):
        self.model = model
        self.with_cfg = with_cfg
        self.stream = torch.cuda.Stream()
        self._pending = None
        self._event = torch.cuda.Event()

    def submit(self, input_ids, pixel_values):
        with torch.cuda.stream(self.stream):
            self._pending = te_forward(self.model, input_ids, pixel_values,
                                       self.with_cfg)
            self._event.record(self.stream)

    def collect(self):
        torch.cuda.current_stream().wait_event(self._event)
        cond, self._pending = self._pending, None
        return cond


# ---------------------------------------------------------------------------
# Functional test
# ---------------------------------------------------------------------------


def build_edit_inputs(tokenizer, model, image, instruction):
    image_processor = model.get_vision_tower().image_processor
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to("cuda", dtype=torch.float16)
    qs = (DEFAULT_IMAGE_TOKEN +
          "\nPlease edit the image based on the following instruction: " +
          instruction)
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
        return_tensors="pt").unsqueeze(0).to("cuda")
    return input_ids, image_tensor


def synthetic_frames(base_image, n):
    """Perturbed copies of the base image standing in for video frames."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    arr = np.asarray(base_image).astype(np.int16)
    frames = []
    for i in range(n):
        shift = rng.integers(-12, 13, size=(1, 1, 3))
        f = np.clip(arr + shift + (i * 2 - n), 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(f))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="checkpoints/Mobile-O-0.5B")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--refresh-every", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out-dir", default="smoke_outputs/streaming")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from PIL import Image
    print("loading model ...", flush=True)
    tokenizer, model, _ = load_pretrained_model(args.model_path)
    model.to("cuda:0")
    model.to(torch.bfloat16)

    base = Image.open("assets/cute_cat.png").convert("RGB")
    frames = synthetic_frames(base, args.frames)
    instruction = "make it look like an oil painting"

    # ---- R1 functional check -------------------------------------------
    print(f"\n=== R1: cond refresh every {args.refresh_every} ===", flush=True)
    cache = CondRefreshCache(model, refresh_every=args.refresh_every)
    per_frame_ms = []
    conds = []
    for i, frame in enumerate(frames):
        input_ids, image_tensor = build_edit_inputs(
            tokenizer, model, frame, instruction)
        torch.cuda.synchronize()
        t0 = time.time()
        cond = cache.get(input_ids, image_tensor)
        torch.cuda.synchronize()
        per_frame_ms.append((time.time() - t0) * 1e3)
        conds.append(cond)
    n_expected = -(-args.frames // args.refresh_every)  # ceil
    print(f"TE calls: {cache.n_te_calls} (expected {n_expected} for "
          f"{args.frames} frames)", flush=True)
    assert cache.n_te_calls == n_expected, "R1 cache not gating TE calls"

    # cached frames return the same object; refresh ticks return new conds
    same_within_window = all(
        conds[i].data_ptr() == conds[i - 1].data_ptr()
        for i in range(1, args.frames) if i % args.refresh_every != 0)
    differ_at_refresh = all(
        not torch.equal(conds[i], conds[i - 1])
        for i in range(1, args.frames) if i % args.refresh_every == 0)
    print(f"cache reuse within window: {same_within_window}", flush=True)
    print(f"cond changes at refresh tick (inputs changed): "
          f"{differ_at_refresh}", flush=True)
    assert same_within_window and differ_at_refresh

    te_ms = [m for i, m in enumerate(per_frame_ms)
             if i % args.refresh_every == 0]
    hit_ms = [m for i, m in enumerate(per_frame_ms)
              if i % args.refresh_every != 0]
    print(f"informal: TE-tick {sum(te_ms)/len(te_ms):.1f} ms vs "
          f"cache-hit {sum(hit_ms)/len(hit_ms):.2f} ms", flush=True)

    # ---- R2 functional check -------------------------------------------
    print("\n=== R2: side-stream TE handoff ===", flush=True)
    input_ids, image_tensor = build_edit_inputs(
        tokenizer, model, frames[0], instruction)
    seq_cond = te_forward(model, input_ids, image_tensor)
    side = SideStreamTE(model)
    side.submit(input_ids, image_tensor)
    side_cond = side.collect()
    match = torch.equal(seq_cond, side_cond)
    print(f"side-stream cond == sequential cond: {match}", flush=True)
    assert match, "R2 stream handoff produced different cond"

    # ---- End-to-end with reduced steps ----------------------------------
    print(f"\n=== E2E: denoise at {args.steps} steps with cached cond ===",
          flush=True)
    torch.cuda.synchronize()
    t0 = time.time()
    imgs = denoise_with_cond(model, conds[-1],
                             num_inference_steps=args.steps)
    torch.cuda.synchronize()
    dt = time.time() - t0
    imgs[0].save(os.path.join(args.out_dir, "r1r2_edit_4step.png"))
    print(f"denoise+decode: {dt*1e3:.0f} ms at {args.steps} steps "
          f"(informal)", flush=True)
    print(f"saved {args.out_dir}/r1r2_edit_4step.png", flush=True)

    print("\nR1R2_FUNCTIONAL_PASS", flush=True)


if __name__ == "__main__":
    main()
