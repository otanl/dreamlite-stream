"""In-pipeline profile: extract per-component CUDA-event timings from a
real champion-config run. Replaces the isolated cold-cache profile with
honest steady-state numbers.

The BatchedEditWorker already records:
  - te_ms (side-stream CUDA event)
  - vae_enc_ms (side-stream CUDA event)
  - denoise_ms (main-stream wall)
  - vae_dec_ms (main-stream wall)
  - total_ms (step_batch_with_prefetch wall)
Side-stream events are async; their elapsed_time captures GPU time on
that stream, not blocking wall time.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402
from dreamlite_lllite.inject import apply_lllite  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from dreamlite_stream import BatchedEditWorker, SharedState  # noqa: E402
from dreamlite_stream.runtime import iter_video_frames  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights", default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--mp4", default=str(_ROOT / "assets" / "davis_mp4" / "blackswan.mp4"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--cond_refresh_every", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_batches", type=int, default=20)
    p.add_argument("--warmup_batches", type=int, default=4)
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--no_lllite", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    controller = None
    if not args.no_lllite:
        print(f"[lllite] {args.lllite_weights}")
        sd = load_file(args.lllite_weights)
        vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
        latent_hw = args.size // vae_downsample
        controller = apply_lllite(
            pipeline.unet, cond_emb_dim=32, mlp_dim=64,
            cond_image_size=args.size, sample_size=latent_hw,
            block_filter=["down_blocks"], inference_mode=True,
            max_batch_size=args.batch_size,
        )
        controller.load_state_dict(sd, strict=False)
        controller.to(device=args.device, dtype=torch.bfloat16)
        controller.eval()
        controller.set_multiplier(1.0)

    # Load enough frames for n_batches; loop if needed
    iterator = iter_video_frames(args.mp4, args.size)
    raw = [f for _, f, _ in iterator]
    if len(raw) < args.batch_size * args.n_batches:
        # Loop
        from itertools import cycle, islice
        raw = list(islice(cycle(raw), args.batch_size * args.n_batches))

    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=args.steps, prompt=args.prompt,
    )
    worker = BatchedEditWorker(
        pipeline=pipeline, state=state, batch_size=args.batch_size,
        device=args.device, dtype=torch.bfloat16, seed=42,
        compile=True, compile_mode="reduce-overhead",
        lllite_controller=controller,
        cond_refresh_every=args.cond_refresh_every,
    )

    print(f"[run] {args.n_batches} batches of {args.batch_size} (warmup {args.warmup_batches})")
    timings = []
    cur_buf = None
    cur_pf = None
    for b in range(args.n_batches):
        buf = raw[b * args.batch_size : (b + 1) * args.batch_size]
        if cur_buf is None:
            cur_buf = buf
            cur_pf = worker.prefetch_batch(cur_buf)
            continue
        nxt_pf = worker.prefetch_batch(buf)
        _, t = worker.step_batch_with_prefetch(cur_buf, cur_pf)
        if b >= args.warmup_batches:
            timings.append(t)
        cur_buf, cur_pf = buf, nxt_pf
    # Final batch
    _, t = worker.step_batch_with_prefetch(cur_buf, cur_pf)
    if args.n_batches - 1 >= args.warmup_batches:
        timings.append(t)

    B = args.batch_size
    n = len(timings)
    print(f"\n[done] {n} measured batches")

    def stats(getter):
        xs = [getter(t) for t in timings]
        return mean(xs), (stdev(xs) if n >= 2 else 0.0)

    te_m, te_s = stats(lambda t: t.te_ms)
    enc_m, enc_s = stats(lambda t: t.vae_enc_ms)
    den_m, den_s = stats(lambda t: t.denoise_ms)
    dec_m, dec_s = stats(lambda t: t.vae_dec_ms)
    tot_m, tot_s = stats(lambda t: t.total_ms)

    fps_total = (B * n) / (sum(t.total_ms for t in timings) / 1000.0)

    print(f"\n========== in-pipeline profile (B={B}, K={args.steps}, ", end="")
    print(f"LLLite={'on (down_blocks)' if controller else 'off'}, refresh={args.cond_refresh_every}) ==========")
    print(f"{'component':<28} {'ms/batch':>14} {'ms/frame':>10}")
    print(f"{'TE (CUDA event, side stream)':<28} {te_m:>7.1f} +- {te_s:<3.1f} {te_m/B:>10.2f}")
    print(f"{'VAE_enc (CUDA event, side)':<28} {enc_m:>7.1f} +- {enc_s:<3.1f} {enc_m/B:>10.2f}")
    print(f"{'UNet denoise (main stream)':<28} {den_m:>7.1f} +- {den_s:<3.1f} {den_m/B:>10.2f}")
    print(f"{'VAE decode (main stream)':<28} {dec_m:>7.1f} +- {dec_s:<3.1f} {dec_m/B:>10.2f}")
    print(f"{'total wall (step_with_prefetch)':<28} {tot_m:>7.1f} +- {tot_s:<3.1f} {tot_m/B:>10.2f}")
    print()
    print(f"throughput: {fps_total:.2f} fps   (1/total_ms = {1000/tot_m * B:.2f} fps)")

    # Interpretation
    side_ms = te_m + enc_m
    main_ms = den_m + dec_m
    print(f"\n[interpretation]")
    print(f"side stream total (TE+VAE_enc):   {side_ms:>7.1f} ms/batch ({side_ms/B:>5.2f} ms/frame)")
    print(f"main stream total (UNet+VAE_dec): {main_ms:>7.1f} ms/batch ({main_ms/B:>5.2f} ms/frame)")
    print(f"step_with_prefetch wall:          {tot_m:>7.1f} ms/batch ({tot_m/B:>5.2f} ms/frame)")
    bottleneck = "side stream (TE-bound)" if side_ms > main_ms else "main stream (UNet+VAE-bound)"
    print(f"bottleneck: {bottleneck}")
    if abs(tot_m - max(side_ms, main_ms)) < 50:
        print(f"  -> wall closely tracks max(side, main); overlap is working well")
    else:
        print(f"  -> wall ({tot_m:.0f}) differs from max(side,main) ({max(side_ms, main_ms):.0f}) by {abs(tot_m - max(side_ms, main_ms)):.0f} ms; overlap is partial")


if __name__ == "__main__":
    main()
