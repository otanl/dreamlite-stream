"""Compare MVP-1 (cond=input only) vs MVP-2 (cond blends keyframe / flow-warped prev).

Produces an mp4 per config under --out_dir, then quality_compare.py can be
run on the directory to compute warping-error metrics. Speed is reported
inline; quality is measured separately via the metrics script.

Run order: all configs run with the same pipeline; the FIRST compile config
triggers torch.compile in place; subsequent configs (compile=False on the
EditWorker) reuse the wrapped UNet.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402

from dreamlite_stream import EditWorker, KeyframeWorker, SharedState  # noqa: E402
from dreamlite_stream.runtime import (  # noqa: E402
    RunStats, run_video, run_video_pipelined,
)
from dreamlite_stream.workers.edit import StepTiming  # noqa: E402


@dataclass
class Config:
    name: str
    w_input: float
    w_prev: float
    w_kf: float
    steps: int = 4
    pipelined: bool = True
    compile_first: bool = False  # only the first compile-target config sets True
    use_keyframe: bool = True


def avg(timings: List[StepTiming], attr: str) -> float:
    return sum(getattr(t, attr) for t in timings) / max(len(timings), 1)


def run_one(
    pipeline,
    cfg: Config,
    args,
    out_dir: Path,
    keyframe_done: List[bool],
) -> Tuple[float, List[StepTiming]]:
    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=cfg.steps, prompt=args.prompt,
    )
    # Always (re-)generate keyframe if this config wants it; cheap (~1s).
    if cfg.use_keyframe and cfg.w_kf > 0:
        kf = KeyframeWorker(
            pipeline=pipeline, state=state, device=args.device,
            dtype=torch.bfloat16, seed=args.seed,
        )
        kf.generate(args.keyframe_prompt or args.prompt)

    worker = EditWorker(
        pipeline=pipeline, state=state, device=args.device, dtype=torch.bfloat16,
        init_mode="pure", seed=args.seed,
        compile=cfg.compile_first, compile_mode=args.compile_mode,
        w_input=cfg.w_input, w_prev=cfg.w_prev, w_kf=cfg.w_kf,
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
    p.add_argument("--prompt", required=True)
    p.add_argument("--keyframe_prompt", default=None)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=30)
    p.add_argument("--compile_mode", default="reduce-overhead")
    p.add_argument("--out_dir", default="out/bench_mvp2")
    p.add_argument("--no_compile", action="store_true",
                   help="skip torch.compile (faster startup, slower per-frame)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    # First config triggers in-place torch.compile of UNet (unless --no_compile).
    first_compile = not args.no_compile
    configs = [
        Config("01_mvp1_baseline_4st",  w_input=1.0, w_prev=0.0, w_kf=0.0, steps=4,
               compile_first=first_compile, use_keyframe=False),
        Config("02_mvp1_baseline_2st",  w_input=1.0, w_prev=0.0, w_kf=0.0, steps=2,
               use_keyframe=False),
        Config("03_mvp2_flow_only",     w_input=0.5, w_prev=0.5, w_kf=0.0, steps=4,
               use_keyframe=False),
        Config("04_mvp2_kf_only",       w_input=0.7, w_prev=0.0, w_kf=0.3, steps=4),
        Config("05_mvp2_balanced",      w_input=0.4, w_prev=0.4, w_kf=0.2, steps=4),
        Config("06_mvp2_strong_prev",   w_input=0.2, w_prev=0.6, w_kf=0.2, steps=4),
        Config("07_mvp2_balanced_2st",  w_input=0.4, w_prev=0.4, w_kf=0.2, steps=2),
    ]

    rows = []
    for cfg in configs:
        print(
            f"\n=== {cfg.name}  "
            f"w=({cfg.w_input},{cfg.w_prev},{cfg.w_kf})  "
            f"steps={cfg.steps} ==="
        )
        fps, measured = run_one(pipeline, cfg, args, out_dir, keyframe_done=[False])
        if not measured:
            continue
        rows.append((cfg, fps, measured))
        last = rows[-1]
        m = last[2]
        print(
            f"  -> wall_fps={fps:.2f}  total={avg(m,'total_ms'):.0f}ms  "
            f"denoise={avg(m,'denoise_ms'):.0f}  te={avg(m,'te_ms'):.0f}"
        )

    print("\n" + "=" * 84)
    print(
        f"{'config':28s}  {'w_in':>4s}  {'w_pr':>4s}  {'w_kf':>4s}  "
        f"{'steps':>5s}  {'fps':>5s}  {'total':>6s}  {'denoise':>7s}"
    )
    print("-" * 84)
    for cfg, fps, m in rows:
        print(
            f"{cfg.name:28s}  {cfg.w_input:4.1f}  {cfg.w_prev:4.1f}  {cfg.w_kf:4.1f}  "
            f"{cfg.steps:5d}  {fps:5.2f}  {avg(m,'total_ms'):6.0f}  {avg(m,'denoise_ms'):7.0f}"
        )

    print(
        f"\nNext: python scripts/quality_compare.py --input {args.input} "
        f"--bench_dir {out_dir} --size {args.size}"
    )


if __name__ == "__main__":
    main()
