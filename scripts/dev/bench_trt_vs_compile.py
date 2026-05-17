"""Head-to-head benchmark: torch.compile UNet vs TRT UNet (base, no LLLite).

Same pipeline / VAE / TE, same DAVIS sequence. The only difference is the UNet
backend. Measures per-batch wall time over N iterations.

Intended for the same-stack 4090 fair comparison: gives us "how much would TRT
buy us if we baked the LLLite hooks in" as an upper bound.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from statistics import mean, stdev

import torch
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402

from dreamlite_stream import BatchedEditWorker, SharedState  # noqa: E402
from dreamlite_stream.runtime import iter_video_frames  # noqa: E402
from dreamlite_stream.trt_unet import TRTUNetWrapper  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--engine", default=str(_ROOT / "out" / "trt" / "unet_b8_512.engine"))
    p.add_argument("--video", default=str(_ROOT / "assets" / "davis_mp4" / "dance-twirl.mp4"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_batches", type=int, default=16,
                   help="batches measured after warmup")
    p.add_argument("--n_warmup", type=int, default=2)
    p.add_argument("--mode", choices=["compile", "trt", "both"], default="both")
    return p.parse_args()


def load_frames(path: str, size: int, n: int):
    out = []
    for idx, frame, _ in iter_video_frames(path, size):
        out.append(frame)
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def bench_worker(pipeline, state, frames, batch_size, n_batches, n_warmup,
                 mode_label: str):
    worker = BatchedEditWorker(
        pipeline=pipeline, state=state, batch_size=batch_size,
        device="cuda", dtype=torch.float16, seed=42,
        compile=(mode_label == "compile"),
        compile_mode="reduce-overhead" if mode_label == "compile" else None,
        lllite_controller=None,
        cond_refresh_every=999,
    )

    total = n_warmup + n_batches
    needed = total * batch_size
    if len(frames) < needed:
        # cycle frames if video too short
        reps = (needed + len(frames) - 1) // len(frames)
        frames = (frames * reps)[:needed]

    timings = []
    for i in range(total):
        buf = frames[i * batch_size: (i + 1) * batch_size]
        _, t = worker.step_batch(buf)
        if i >= n_warmup:
            timings.append(t)

    n_meas = sum(t.n_frames for t in timings)
    sum_total = sum(t.total_ms for t in timings)
    fps = n_meas / (sum_total / 1000)
    per_batch_ms = [t.total_ms for t in timings]
    return {
        "mode": mode_label,
        "fps": fps,
        "n_batches": len(timings),
        "n_frames": n_meas,
        "per_batch_ms_mean": mean(per_batch_ms),
        "per_batch_ms_std": stdev(per_batch_ms) if len(per_batch_ms) > 1 else 0.0,
        "te_ms_mean": mean(t.te_ms for t in timings),
        "denoise_ms_mean": mean(t.denoise_ms for t in timings),
        "vae_enc_ms_mean": mean(t.vae_enc_ms for t in timings),
        "vae_dec_ms_mean": mean(t.vae_dec_ms for t in timings),
    }


def main():
    args = parse_args()

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.float16,
    ).to("cuda")
    pyt_unet = pipeline.unet

    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=1, prompt=args.prompt,
    )

    print(f"[frames] reading {args.video}")
    frames = load_frames(args.video, args.size, (args.n_warmup + args.n_batches) * args.batch_size)

    results = []

    if args.mode in ("compile", "both"):
        print("\n" + "=" * 60)
        print("[mode] torch.compile (eager UNet + reduce-overhead)")
        print("=" * 60)
        pipeline.unet = pyt_unet
        r = bench_worker(pipeline, state, frames, args.batch_size,
                         args.n_batches, args.n_warmup, "compile")
        results.append(r)
        print(f"  fps = {r['fps']:.2f}  per-batch {r['per_batch_ms_mean']:.1f}±{r['per_batch_ms_std']:.1f} ms")

    if args.mode in ("trt", "both"):
        print("\n" + "=" * 60)
        print(f"[mode] TRT  engine={args.engine}")
        print("=" * 60)
        engine_path = Path(args.engine)
        if not engine_path.exists():
            raise SystemExit(f"engine not found: {engine_path}")
        pipeline.unet = TRTUNetWrapper(args.engine, device="cuda")
        r = bench_worker(pipeline, state, frames, args.batch_size,
                         args.n_batches, args.n_warmup, "trt")
        results.append(r)
        print(f"  fps = {r['fps']:.2f}  per-batch {r['per_batch_ms_mean']:.1f}±{r['per_batch_ms_std']:.1f} ms")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'mode':<10}  {'fps':>7}  {'batch_ms':>14}  {'te_ms':>8}  {'denoise_ms':>10}")
    for r in results:
        print(f"{r['mode']:<10}  {r['fps']:>7.2f}  "
              f"{r['per_batch_ms_mean']:>6.1f}±{r['per_batch_ms_std']:<5.1f}  "
              f"{r['te_ms_mean']:>8.1f}  {r['denoise_ms_mean']:>10.1f}")
    if len(results) == 2:
        speedup = results[1]["fps"] / results[0]["fps"]
        print(f"\nTRT vs torch.compile speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
