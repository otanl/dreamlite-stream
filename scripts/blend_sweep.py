"""Sweep post-hoc output_blend alpha values on the MVP-1 best speed config.

Holds the model + worker config fixed at compile + pipelined + 2-step (the
MVP-1.5 speed-quality champion) and varies only the post-decode blend alpha.

Outputs go to --out_dir; run quality_compare.py afterwards.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402

from dreamlite_stream import EditWorker, SharedState  # noqa: E402
from dreamlite_stream.output_blend import OutputBlender  # noqa: E402
from dreamlite_stream.runtime import run_video_pipelined  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--input", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=30)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[1.0, 0.85, 0.7, 0.5, 0.3])
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--out_dir", default="out/bench_blend")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    rows = []
    for i, alpha in enumerate(args.alphas):
        cfg_name = f"a{alpha:.2f}".replace(".", "")
        first_compile = (i == 0) and not args.no_compile

        state = SharedState(
            height=args.size, width=args.size,
            num_inference_steps=args.steps, prompt=args.prompt,
        )
        worker = EditWorker(
            pipeline=pipeline, state=state, device=args.device,
            dtype=torch.bfloat16, init_mode="pure", seed=args.seed,
            compile=first_compile, compile_mode="reduce-overhead",
        )
        blender = OutputBlender(alpha=alpha)

        out_path = str(out_dir / f"blend_{cfg_name}.mp4")
        total = args.warmup + args.measure
        print(f"\n=== alpha={alpha:.2f} ===")
        stats = run_video_pipelined(
            worker, in_path=args.input, out_path=out_path,
            size=args.size, max_frames=total, log_every=0, blender=blender,
        )
        measured = stats.timings[args.warmup:]
        if not measured:
            continue
        wall_ms = sum(t.total_ms for t in measured)
        fps = len(measured) / (wall_ms / 1000)
        avg_total = wall_ms / len(measured)
        rows.append((alpha, fps, avg_total))
        print(f"  -> wall_fps={fps:.2f}  total={avg_total:.0f}ms")

    print("\n" + "=" * 50)
    print(f"{'alpha':>6s}  {'fps':>5s}  {'total':>6s}")
    print("-" * 50)
    for alpha, fps, tot in rows:
        print(f"{alpha:6.2f}  {fps:5.2f}  {tot:6.0f}")

    print(
        f"\nNext: python scripts/quality_compare.py --input {args.input} "
        f"--bench_dir {out_dir} --size {args.size}"
    )


if __name__ == "__main__":
    main()
