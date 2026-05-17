"""Sustained throughput + end-to-end latency test, with Dynamo cache
sized large enough to avoid recompile thrashing on long runs.

Setting `torch._dynamo.config.cache_size_limit = 256` (vs default 8)
should keep all compiled variants in cache for the duration of a long
streaming run.

Reports:
  - mean fps, std
  - per-batch wall: p50, p95, max
  - per-frame end-to-end latency (batch buffering + processing):
    p50, p95 at a virtual 30 fps source
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import cycle, islice
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

# Bump Dynamo cache before any torch.compile triggers
import torch._dynamo  # noqa: E402
torch._dynamo.config.cache_size_limit = 256

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from PIL import Image  # noqa: E402

from dreamlite import DreamLiteMobilePipeline  # noqa: E402
from dreamlite_lllite.inject import apply_lllite  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from dreamlite_stream import BatchedEditWorker, SharedState  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights", default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--mp4", default=str(_ROOT / "assets" / "davis_mp4" / "parkour.mp4"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--total_frames", type=int, default=480)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--cond_refresh_every", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup_batches", type=int, default=4)
    p.add_argument("--source_fps", type=float, default=30.0, help="virtual source fps for latency calculation")
    return p.parse_args()


def percentile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    f, c = int(k), int(k) + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])


@torch.no_grad()
def main():
    args = parse_args()
    print(f"[setup] Dynamo cache_size_limit = {torch._dynamo.config.cache_size_limit}")

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

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

    # Load and loop frames
    import cv2
    cap = cv2.VideoCapture(args.mp4)
    raw = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f_rgb = cv2.resize(f_rgb, (args.size, args.size), interpolation=cv2.INTER_AREA)
        raw.append(Image.fromarray(f_rgb))
    cap.release()
    if not raw:
        raise RuntimeError(f"empty mp4 {args.mp4}")
    frames = list(islice(cycle(raw), args.total_frames))
    print(f"[frames] {len(frames)} frames (looped from {len(raw)})")

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

    print(f"[run] {len(frames) // args.batch_size} batches, B={args.batch_size}, warmup={args.warmup_batches}")
    batch_walls = []
    cur_buf = None
    cur_pf = None
    n_batches = len(frames) // args.batch_size
    for b in range(n_batches):
        buf = frames[b * args.batch_size : (b + 1) * args.batch_size]
        if cur_buf is None:
            cur_buf = buf
            cur_pf = worker.prefetch_batch(cur_buf)
            continue
        nxt_pf = worker.prefetch_batch(buf)
        t0 = time.perf_counter()
        _, t = worker.step_batch_with_prefetch(cur_buf, cur_pf)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if b >= args.warmup_batches:
            batch_walls.append(dt)
        cur_buf, cur_pf = buf, nxt_pf
    # Final batch
    t0 = time.perf_counter()
    _, t = worker.step_batch_with_prefetch(cur_buf, cur_pf)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    batch_walls.append(dt)

    n = len(batch_walls)
    B = args.batch_size
    mean_wall = mean(batch_walls)
    fps = (n * B) / (sum(batch_walls) / 1000.0)

    p50 = percentile(batch_walls, 0.50)
    p95 = percentile(batch_walls, 0.95)
    p99 = percentile(batch_walls, 0.99)
    max_wall = max(batch_walls)

    # End-to-end latency at virtual source fps
    source_dt_ms = 1000.0 / args.source_fps
    buffering_ms = (B - 1) * source_dt_ms
    latency_p50_ms = buffering_ms + p50
    latency_p95_ms = buffering_ms + p95

    print(f"\n========== sustained throughput (B={B}, K={args.steps}, {n} measured batches) ==========")
    print(f"throughput:     {fps:>7.2f} fps  ({n*B} frames in {sum(batch_walls)/1000:.1f}s)")
    print(f"per-batch wall: mean={mean_wall:>6.1f}ms, std={stdev(batch_walls) if n>=2 else 0:.1f}ms")
    print(f"                p50={p50:>6.1f}ms, p95={p95:>6.1f}ms, p99={p99:>6.1f}ms, max={max_wall:>6.1f}ms")

    print(f"\n========== end-to-end latency at {args.source_fps:.0f} fps source ==========")
    print(f"buffering (B-1 frames @ {args.source_fps:.0f} fps): {buffering_ms:>6.1f}ms")
    print(f"end-to-end p50: {latency_p50_ms:>6.1f}ms")
    print(f"end-to-end p95: {latency_p95_ms:>6.1f}ms")

    out = Path(_ROOT) / "out" / "sustained_throughput.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {
            "batch_size": B, "steps": args.steps,
            "total_frames": args.total_frames, "warmup_batches": args.warmup_batches,
            "cache_size_limit": torch._dynamo.config.cache_size_limit,
            "source_fps": args.source_fps,
        },
        "throughput_fps": fps,
        "batch_wall_ms": {
            "mean": mean_wall, "std": stdev(batch_walls) if n >= 2 else 0.0,
            "p50": p50, "p95": p95, "p99": p99, "max": max_wall,
        },
        "latency_ms": {
            "buffering": buffering_ms,
            "e2e_p50": latency_p50_ms, "e2e_p95": latency_p95_ms,
        },
    }, indent=2), encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
