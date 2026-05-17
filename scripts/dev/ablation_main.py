"""Ablation study across 10 DAVIS sequences and 5 methods.

Methods:
  base_4st          : MVP-1 baseline 4-step (compile triggered here)
  base_2st          : MVP-1 baseline 2-step
  blend_2st         : MVP-1.5 champion (compile + pipelined + 2-step + blend 0.85)
  lllite            : LLLite alone (best quality, 1.34 FPS)
  spec_t50          : LLLite + speculative (flow_thresh=50, max_consec=4)
  spec_t200         : LLLite + speculative (loose threshold; max_consec dominates)

For each sequence × method we record:
  - measured FPS (per-step total time)
  - mean warping_error vs input
  - hit rate (for spec methods)
  - frame count

Aggregated outputs:
  results.jsonl     : raw per-sequence-per-method rows
  summary.csv       : pivoted table (rows = method, cols = sequence)
  pareto.csv        : speed/quality scatter for plotting
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch._dynamo
from safetensors.torch import load_file

torch._dynamo.config.cache_size_limit = 64

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
_LLLITE = _ROOT.parent / "dreamlite-lllite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))
sys.path.insert(0, str(_LLLITE / "src"))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402
from dreamlite_lllite import apply_lllite  # noqa: E402

from dreamlite_stream import (  # noqa: E402
    EditWorker, SharedState, SpeculativeEditWorker,
)
from dreamlite_stream.metrics import compute_temporal, read_video_frames  # noqa: E402
from dreamlite_stream.output_blend import OutputBlender  # noqa: E402
from dreamlite_stream.runtime import VideoWriter, iter_video_frames  # noqa: E402


@dataclass
class MethodResult:
    sequence: str
    method: str
    n_frames: int
    fps: float
    avg_total_ms: float
    avg_denoise_ms: float
    warp_err: float
    consecutive_l1: float
    consistency_ratio: float
    hit_rate: float = 0.0
    extra: Dict[str, float] = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights",
                   default=str(_ROOT / "runs" / "temporal_lllite_v1" / "temporal_lllite_step000420.safetensors"))
    p.add_argument("--sequences", nargs="+", default=[
        "blackswan", "libby", "swing", "camel", "dance-twirl",
        "goat", "scooter-black", "bmx-trees", "parkour", "kite-surf",
    ])
    p.add_argument("--mp4_dir", default=str(_ROOT / "assets" / "davis_mp4"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=30)
    p.add_argument("--out_dir", default=str(_ROOT / "out" / "ablation"))
    return p.parse_args()


def stylize_with_method(
    pipeline, controller, args, in_path: str, method: str, out_path: str,
) -> tuple[List, float, float, int]:
    """Run one method on one video. Returns (timings, fps, hit_rate, n_frames)."""
    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=4 if "4st" in method else 2 if "2st" in method else 4,
        prompt=args.prompt,
    )
    blender = None
    use_spec = False
    spec_thresh = 0.0

    # We compile once on the first method that requests it; subsequent
    # methods inherit the wrapped UNet. The order in main() is important.
    if method == "base_4st":
        worker = EditWorker(
            pipeline=pipeline, state=state, device=args.device,
            dtype=torch.bfloat16, init_mode="pure", seed=args.seed,
            compile=True, lllite_controller=None,
        )
    elif method == "base_2st":
        worker = EditWorker(
            pipeline=pipeline, state=state, device=args.device,
            dtype=torch.bfloat16, init_mode="pure", seed=args.seed,
            compile=False, lllite_controller=None,
        )
    elif method == "blend_2st":
        worker = EditWorker(
            pipeline=pipeline, state=state, device=args.device,
            dtype=torch.bfloat16, init_mode="pure", seed=args.seed,
            compile=False, lllite_controller=None,
        )
        blender = OutputBlender(alpha=0.85)
    elif method == "lllite":
        controller.set_multiplier(1.0)
        worker = EditWorker(
            pipeline=pipeline, state=state, device=args.device,
            dtype=torch.bfloat16, init_mode="pure", seed=args.seed,
            compile=False, lllite_controller=controller,
        )
    elif method.startswith("spec_t"):
        thr = float(method.replace("spec_t", ""))
        controller.set_multiplier(1.0)
        inner = EditWorker(
            pipeline=pipeline, state=state, device=args.device,
            dtype=torch.bfloat16, init_mode="pure", seed=args.seed,
            compile=False, lllite_controller=controller,
        )
        worker = SpeculativeEditWorker(inner=inner, flow_thresh=thr, max_consec=4)
        use_spec = True
    else:
        raise ValueError(f"unknown method: {method}")

    writer = None
    timings = []
    n_hits = 0
    total_seq = args.warmup + args.measure

    for idx, frame, fps in iter_video_frames(in_path, args.size):
        if idx >= total_seq:
            break
        if blender is not None:
            from PIL import Image
            out_pil, t = worker.step(frame)
            in_rgb = np.asarray(frame.convert("RGB"))
            out_rgb = np.asarray(out_pil.convert("RGB"))
            out_rgb = blender.apply(out_rgb, in_rgb)
            out_pil = Image.fromarray(out_rgb)
        else:
            out_pil, t = worker.step(frame)

        if writer is None:
            writer = VideoWriter(out_path, args.size, fps)
        writer.write_pil(out_pil)
        timings.append(t)
        if use_spec and getattr(t, "accepted", False):
            n_hits += 1

    if writer:
        writer.close()

    measured = timings[args.warmup:]
    if not measured:
        return [], 0.0, 0.0, 0

    wall_ms = sum(t.total_ms for t in measured)
    fps = len(measured) / (wall_ms / 1000)
    hit_rate = (
        sum(1 for t in measured if getattr(t, "accepted", False)) / len(measured)
        if use_spec else 0.0
    )
    return measured, fps, hit_rate, len(measured)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_dir = Path(args.mp4_dir)

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    # Attach LLLite once.
    vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
    latent_hw = args.size // vae_downsample
    print(f"[lllite] attaching {args.lllite_weights}")
    controller = apply_lllite(
        pipeline.unet, cond_emb_dim=32, mlp_dim=64,
        cond_image_size=args.size, sample_size=latent_hw,
        inference_mode=True,
    )
    controller.load_state_dict(load_file(args.lllite_weights), strict=True)
    controller.to(device=args.device, dtype=torch.bfloat16)
    controller.eval()
    controller.set_multiplier(0.0)  # disabled by default; method enables when needed

    methods = ["base_4st", "base_2st", "blend_2st", "lllite", "spec_t50", "spec_t200"]

    results: List[MethodResult] = []
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    for seq in args.sequences:
        in_path = mp4_dir / f"{seq}.mp4"
        if not in_path.exists():
            print(f"  skip {seq}: missing {in_path}")
            continue
        print(f"\n=== {seq} ===")
        # Need the input frames for warping_error metric.
        in_frames = read_video_frames(str(in_path), size=args.size)

        for method in methods:
            seq_out_dir = out_dir / "videos" / seq
            seq_out_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(seq_out_dir / f"{method}.mp4")

            controller.set_multiplier(0.0)  # reset before each run
            t_method = time.perf_counter()
            timings, fps, hit_rate, n_meas = stylize_with_method(
                pipeline, controller, args, str(in_path), method, out_path,
            )
            elapsed = time.perf_counter() - t_method
            if n_meas == 0:
                print(f"  {method:14s}  (skipped, no frames)")
                continue

            # Compute quality metrics by re-reading the output mp4.
            out_frames = read_video_frames(out_path, size=None)
            n = min(len(in_frames), len(out_frames))
            if n >= 2:
                m = compute_temporal(in_frames[:n], out_frames[:n])
            else:
                m = None

            avg_total = sum(t.total_ms for t in timings) / len(timings)
            avg_denoise = sum(t.denoise_ms for t in timings) / len(timings)
            r = MethodResult(
                sequence=seq, method=method, n_frames=n_meas, fps=fps,
                avg_total_ms=avg_total, avg_denoise_ms=avg_denoise,
                warp_err=(m.warping_error if m else 0.0),
                consecutive_l1=(m.consecutive_l1 if m else 0.0),
                consistency_ratio=(m.consistency_ratio if m else 0.0),
                hit_rate=hit_rate, extra={"wall_s": float(elapsed)},
            )
            results.append(r)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r.__dict__) + "\n")
            print(
                f"  {method:14s}  fps={fps:5.2f}  total={avg_total:6.0f}ms  "
                f"warp_err={r.warp_err:6.2f}  hit={hit_rate*100:4.0f}%"
            )

    # Aggregate per-method statistics.
    print("\n" + "=" * 90)
    print(f"{'method':15s}  {'avg_fps':>8s}  {'avg_warp_err':>12s}  {'avg_hit':>8s}  N")
    print("-" * 90)
    by_method: Dict[str, List[MethodResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)
    for method in methods:
        rs = by_method.get(method, [])
        if not rs:
            continue
        avg_fps = sum(r.fps for r in rs) / len(rs)
        avg_we = sum(r.warp_err for r in rs) / len(rs)
        avg_hit = sum(r.hit_rate for r in rs) / len(rs)
        print(f"{method:15s}  {avg_fps:8.2f}  {avg_we:12.2f}  {avg_hit*100:7.0f}%  {len(rs)}")

    # Save Pareto CSV.
    pareto_path = out_dir / "pareto.csv"
    with pareto_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "method", "fps", "warp_err", "consistency_ratio", "hit_rate"])
        for r in results:
            w.writerow([r.sequence, r.method, f"{r.fps:.3f}",
                        f"{r.warp_err:.3f}", f"{r.consistency_ratio:.3f}",
                        f"{r.hit_rate:.3f}"])
    print(f"\n[saved] {pareto_path}")


if __name__ == "__main__":
    main()
