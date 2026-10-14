"""Sustained throughput + end-to-end latency test, with Dynamo cache
sized large enough to avoid recompile thrashing on long runs.

Setting `torch._dynamo.config.cache_size_limit = 256` (vs default 8)
should keep all compiled variants in cache for the duration of a long
streaming run.

Each loop iteration is `prefetch_batch(next)` followed by
`step_batch_with_prefetch(cur)`, both on the calling thread. Two rates are
reported:

  - loop fps:      frames / sum(iteration walls). Prefetch + step; this is
                   what a caller of the loop gets.
  - step-only fps: frames / sum(step walls). The step half alone. Earlier
                   versions of this script reported only this figure (as
                   `throughput_fps` / `batch_wall_ms`); it leaves out the time
                   the thread spends inside `prefetch_batch`.

Pass --lean (or the individual flags it stands for; see
dreamlite_stream/lean.py) to measure the reduced per-batch path, and --size
to change the resolution. The defaults reproduce the paper's configuration.

Reports:
  - loop fps and step-only fps
  - per-iteration wall: p50, p95, max, and the prefetch / step halves
  - per-frame end-to-end latency (batch buffering + iteration wall):
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
_LLLITE = _ROOT.parent / "dreamlite-lllite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))
sys.path.insert(0, str(_LLLITE / "src"))

warnings.filterwarnings("ignore")

from PIL import Image  # noqa: E402

from dreamlite import DreamLiteMobilePipeline  # noqa: E402
from dreamlite_lllite.inject import apply_lllite  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from dreamlite_stream import BatchedEditWorker, SharedState  # noqa: E402
from dreamlite_stream.lean import (  # noqa: E402
    add_lean_args, apply_onestep_lora, apply_trt_engine, lean_worker_kwargs, resolve_lean_args,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights", default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--mp4", default=str(_ROOT / "assets" / "davis_mp4" / "parkour.mp4"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--total_frames", type=int, default=480)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--cond_refresh_every", type=int, default=None,
                   help="Refresh the LLLite cond every N batches. 0 = set it "
                        "once and hold. Default 8, or 0 with --lean.")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup_batches", type=int, default=4)
    p.add_argument("--source_fps", type=float, default=30.0, help="virtual source fps for latency calculation")
    p.add_argument("--out_json", default=str(_ROOT / "out" / "sustained_throughput.json"))
    add_lean_args(p)
    return resolve_lean_args(p.parse_args())


def percentile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    f, c = int(k), int(k) + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])


def wall_stats(xs):
    return {
        "mean": mean(xs), "std": stdev(xs) if len(xs) >= 2 else 0.0,
        "p50": percentile(xs, 0.50), "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99), "max": max(xs),
    }


@torch.no_grad()
def main():
    args = parse_args()
    print(f"[setup] Dynamo cache_size_limit = {torch._dynamo.config.cache_size_limit}")

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)
    apply_onestep_lora(pipeline, args)

    print(f"[lllite] {args.lllite_weights}")
    sd = load_file(args.lllite_weights)
    vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
    latent_hw = args.size // vae_downsample
    if args.lllite_multiplier != 0:
        controller = apply_lllite(
            pipeline.unet, cond_emb_dim=32, mlp_dim=64,
            cond_image_size=args.size, sample_size=latent_hw,
            block_filter=["down_blocks"], inference_mode=True,
            max_batch_size=args.batch_size,
        )
        controller.load_state_dict(sd, strict=False)
        controller.to(device=args.device, dtype=torch.bfloat16)
        controller.eval()
        controller.set_multiplier(args.lllite_multiplier)
    else:
        # Multiplier 0 means no adapter: attach nothing, rather than hooks that add zero.
        controller = None
        print("[lllite] off (--lllite_multiplier 0): adapter not attached")
    apply_trt_engine(pipeline, args, controller)

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
        # The fast VAE path takes uint8 arrays directly, as a capture source
        # would deliver them; the default path takes PIL images.
        raw.append(f_rgb if args.fast_vae else Image.fromarray(f_rgb))
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
        device=args.device, dtype=torch.bfloat16, seed=args.seed,
        compile=True, compile_mode="reduce-overhead",
        lllite_controller=controller,
        **lean_worker_kwargs(args),
    )
    print(
        f"[config] size={args.size} B={args.batch_size} K={args.steps}  "
        f"te_refresh_every={args.te_refresh_every} te_batch_one={args.te_batch_one}  "
        f"cond_refresh_every={args.cond_refresh_every}  shared_noise={args.shared_noise}  "
        f"fast_vae={args.fast_vae} vae_compile={args.vae_compile}  "
        f"onestep_lora={'yes' if args.onestep_lora else 'no'} lllite_multiplier={args.lllite_multiplier}  "
        f"trt_engine={'yes' if args.trt_engine else 'no'}"
    )

    print(f"[run] {len(frames) // args.batch_size} batches, B={args.batch_size}, warmup={args.warmup_batches}")
    # One iteration = prefetch_batch(next) then step(cur), both on this thread.
    # The side stream overlaps TE/VAE-encode GPU work with the step, but the
    # prefetch call itself (tokenise, image preprocessing, and a TE dispatch
    # that blocks on host syncs) holds this thread before the step starts.
    prefetch_walls = []
    step_walls = []
    iter_walls = []
    cur_buf = None
    cur_pf = None
    n_batches = len(frames) // args.batch_size
    for b in range(n_batches):
        buf = frames[b * args.batch_size : (b + 1) * args.batch_size]
        if cur_buf is None:
            cur_buf = buf
            cur_pf = worker.prefetch_batch(cur_buf)
            continue
        t_it = time.perf_counter()
        nxt_pf = worker.prefetch_batch(buf)
        t0 = time.perf_counter()
        _, t = worker.step_batch_with_prefetch(cur_buf, cur_pf)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if b >= args.warmup_batches:
            prefetch_walls.append((t0 - t_it) * 1000)
            step_walls.append((t1 - t0) * 1000)
            iter_walls.append((t1 - t_it) * 1000)
        cur_buf, cur_pf = buf, nxt_pf
    # Final batch: nothing is left to prefetch, so this is a step without the
    # prefetch half. Run it, but keep it out of the per-iteration statistics.
    worker.step_batch_with_prefetch(cur_buf, cur_pf)
    torch.cuda.synchronize()

    if not iter_walls:
        raise RuntimeError("no measured iterations; raise --total_frames or lower --warmup_batches")

    n = len(iter_walls)
    B = args.batch_size
    loop_fps = (n * B) / (sum(iter_walls) / 1000.0)
    step_fps = (n * B) / (sum(step_walls) / 1000.0)
    it, st, pf = wall_stats(iter_walls), wall_stats(step_walls), wall_stats(prefetch_walls)

    # End-to-end latency at virtual source fps: a frame waits for its batch to
    # fill, then for one full iteration.
    source_dt_ms = 1000.0 / args.source_fps
    buffering_ms = (B - 1) * source_dt_ms
    latency_p50_ms = buffering_ms + it["p50"]
    latency_p95_ms = buffering_ms + it["p95"]

    print(f"\n========== sustained throughput (B={B}, K={args.steps}, {n} measured iterations) ==========")
    print(f"loop:      {loop_fps:>7.2f} fps  prefetch + step ({n*B} frames in {sum(iter_walls)/1000:.1f}s)")
    print(f"step-only: {step_fps:>7.2f} fps  step half alone; excludes the prefetch call")
    print(f"per-iteration wall: mean={it['mean']:>6.1f}ms, std={it['std']:.1f}ms")
    print(f"                    p50={it['p50']:>6.1f}ms, p95={it['p95']:>6.1f}ms, p99={it['p99']:>6.1f}ms, max={it['max']:>6.1f}ms")
    print(f"  prefetch half:    p50={pf['p50']:>6.1f}ms, p95={pf['p95']:>6.1f}ms")
    print(f"  step half:        p50={st['p50']:>6.1f}ms, p95={st['p95']:>6.1f}ms")

    print(f"\n========== end-to-end latency at {args.source_fps:.0f} fps source ==========")
    print(f"buffering (B-1 frames @ {args.source_fps:.0f} fps): {buffering_ms:>6.1f}ms")
    print(f"end-to-end p50: {latency_p50_ms:>6.1f}ms")
    print(f"end-to-end p95: {latency_p95_ms:>6.1f}ms")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {
            "batch_size": B, "steps": args.steps, "size": args.size,
            "total_frames": args.total_frames, "warmup_batches": args.warmup_batches,
            "cond_refresh_every": args.cond_refresh_every,
            "te_refresh_every": args.te_refresh_every, "te_batch_one": args.te_batch_one,
            "shared_noise": args.shared_noise,
            "fast_vae": args.fast_vae, "vae_compile": args.vae_compile,
            "onestep_lora": args.onestep_lora, "lllite_multiplier": args.lllite_multiplier,
            "trt_engine": args.trt_engine,
            "cache_size_limit": torch._dynamo.config.cache_size_limit,
            "source_fps": args.source_fps,
        },
        "loop_fps": loop_fps,
        "step_only_fps": step_fps,
        "iter_wall_ms": it,
        "prefetch_wall_ms": pf,
        "step_wall_ms": st,
        "latency_ms": {
            "buffering": buffering_ms,
            "e2e_p50": latency_p50_ms, "e2e_p95": latency_p95_ms,
        },
    }, indent=2), encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
