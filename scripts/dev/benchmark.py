"""Benchmark configurations of EditWorker side-by-side on the same input clip.

Run order matters: torch.compile mutates pipeline.unet in place, so the
non-compile configs run first.

Usage:
    python scripts/benchmark.py --input assets/test_clip.mp4 --size 512 \
        --warmup 3 --measure 12
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402

from dreamlite_stream import EditWorker, SharedState  # noqa: E402
from dreamlite_stream.runtime import (  # noqa: E402
    RunStats,
    run_video,
    run_video_pipelined,
)
from dreamlite_stream.workers.edit import StepTiming  # noqa: E402


@dataclass
class Config:
    name: str
    steps: int
    pipelined: bool
    compile: bool
    init_mode: str = "pure"
    noise_strength: float = 0.7


def avg(timings: List[StepTiming], attr: str) -> float:
    if not timings:
        return 0.0
    return sum(getattr(t, attr) for t in timings) / len(timings)


def run_one(
    pipeline,
    cfg: Config,
    args,
    out_dir: Path,
) -> Tuple[float, List[StepTiming]]:
    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=cfg.steps, prompt=args.prompt,
    )
    worker = EditWorker(
        pipeline=pipeline, state=state, device=args.device,
        dtype=torch.bfloat16,
        init_mode=cfg.init_mode, noise_strength=cfg.noise_strength,
        seed=args.seed,
        compile=cfg.compile, compile_mode=args.compile_mode,
    )
    runner = run_video_pipelined if cfg.pipelined else run_video
    out_path = str(out_dir / f"{cfg.name}.mp4")
    total = args.warmup + args.measure
    stats: RunStats = runner(
        worker, in_path=args.input, out_path=out_path,
        size=args.size, max_frames=total, log_every=0,
    )
    measured = stats.timings[args.warmup:]
    if not measured:
        return 0.0, []
    wall_ms = sum(t.total_ms for t in measured)
    fps = len(measured) / (wall_ms / 1000)
    return fps, measured


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--input", required=True)
    p.add_argument("--prompt", default="transfer this to oil painting style")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=12)
    p.add_argument("--compile_mode", default="reduce-overhead")
    p.add_argument("--out_dir", default="out/bench")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    # IMPORTANT: compile=True wraps pipeline.unet in place. Run all non-compile
    # configs first; the *first* compile config does the wrap; subsequent
    # configs reuse the already-compiled unet (compile=False on EditWorker).
    # Order matters: non-compile configs first; first compile config does the
    # in-place wrap; subsequent compile configs reuse the wrapped unet.
    configs = [
        Config("01_baseline_pure",           steps=4, pipelined=False, compile=False, init_mode="pure"),
        Config("02_baseline_prev",           steps=4, pipelined=False, compile=False, init_mode="prev"),
        Config("03_compile_setup",           steps=4, pipelined=False, compile=True,  init_mode="pure"),
        Config("04_compile_pipe_pure_4st",   steps=4, pipelined=True,  compile=False, init_mode="pure"),
        Config("05_compile_pipe_prev_4st",   steps=4, pipelined=True,  compile=False, init_mode="prev"),
        Config("06_compile_pipe_pure_2st",   steps=2, pipelined=True,  compile=False, init_mode="pure"),
        Config("07_compile_pipe_prev_2st",   steps=2, pipelined=True,  compile=False, init_mode="prev"),
    ]

    rows = []
    for cfg in configs:
        print(
            f"\n=== {cfg.name}  steps={cfg.steps} pipelined={cfg.pipelined} "
            f"compile={cfg.compile} init={cfg.init_mode} ==="
        )
        fps, measured = run_one(pipeline, cfg, args, out_dir)
        if not measured:
            continue
        rows.append((
            cfg.name, fps,
            avg(measured, "total_ms"),
            avg(measured, "te_ms"),
            avg(measured, "vae_enc_ms"),
            avg(measured, "denoise_ms"),
            avg(measured, "vae_dec_ms"),
        ))
        last = rows[-1]
        print(
            f"  -> wall_fps={last[1]:.2f}  total={last[2]:.0f}ms  "
            f"te={last[3]:.0f}  enc={last[4]:.0f}  "
            f"denoise={last[5]:.0f}  dec={last[6]:.0f}"
        )

    print("\n" + "=" * 78)
    print(f"size={args.size}x{args.size}  warmup={args.warmup}  measure={args.measure}")
    print(
        f"{'config':34s}  {'fps':>6s}  {'total':>6s}  {'te':>5s}  "
        f"{'enc':>4s}  {'denoise':>7s}  {'dec':>4s}"
    )
    print("-" * 80)
    base = rows[0][1] if rows else 1.0
    for name, fps, tot, te, enc, den, dec in rows:
        speedup = fps / base if base > 0 else 0.0
        print(
            f"{name:34s}  {fps:6.2f}  {tot:6.0f}  {te:5.0f}  "
            f"{enc:4.0f}  {den:7.0f}  {dec:4.0f}   ({speedup:.2f}x)"
        )


if __name__ == "__main__":
    main()
