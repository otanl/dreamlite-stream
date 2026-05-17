"""Compare temporal-consistency metrics across configs produced by benchmark.py.

Usage:
    python scripts/quality_compare.py \
        --input assets/davis/horsejump-low.mp4 \
        --bench_dir out/bench --size 512 \
        --grid_frame 10 --grid_out out/bench/grid.png

For each *.mp4 in --bench_dir, computes:
  warping_error    : photometric L1 of out_t vs flow-warped out_{t+1}
  consecutive_l1   : raw L1 of out_t vs out_{t+1}
  consistency      : warping_error / consecutive_l1
The lower warping_error, the more the output respects input motion (= less flicker).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dreamlite_stream.metrics import (  # noqa: E402
    TemporalMetrics,
    compute_temporal,
    make_grid,
    read_video_frames,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="reference input video")
    p.add_argument("--bench_dir", default="out/bench", help="dir of benchmark *.mp4 outputs")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--max_frames", type=int, default=64,
                   help="cap to this many frames from the start (for speed)")
    p.add_argument("--grid_frame", type=int, default=10,
                   help="frame index to use in side-by-side grid")
    p.add_argument("--grid_out", default="out/bench/grid.png")
    return p.parse_args()


def main():
    args = parse_args()
    bench_dir = Path(args.bench_dir)
    if not bench_dir.is_dir():
        raise FileNotFoundError(bench_dir)

    print(f"[input] {args.input}  size={args.size}  max_frames={args.max_frames}")
    in_frames = read_video_frames(args.input, size=args.size)[: args.max_frames]
    print(f"  loaded {len(in_frames)} frames")

    out_files = sorted(bench_dir.glob("*.mp4"))
    if not out_files:
        raise FileNotFoundError(f"no *.mp4 in {bench_dir}")

    rows: List[Tuple[str, TemporalMetrics]] = []
    grid_panels: List[Tuple[str, np.ndarray]] = []

    # Reference panel: input frame
    if 0 <= args.grid_frame < len(in_frames):
        grid_panels.append(("input", cv2.cvtColor(in_frames[args.grid_frame], cv2.COLOR_RGB2BGR)))

    for f in out_files:
        out_frames = read_video_frames(str(f), size=None)[: args.max_frames]
        # Some outputs may be shorter (e.g. when benchmark capped frames).
        n = min(len(in_frames), len(out_frames))
        if n < 2:
            print(f"  skip {f.name}: only {n} frames")
            continue
        m = compute_temporal(in_frames[:n], out_frames[:n])
        rows.append((f.stem, m))
        if 0 <= args.grid_frame < n:
            label = f.stem.replace("_", " ")
            grid_panels.append((label, cv2.cvtColor(out_frames[args.grid_frame], cv2.COLOR_RGB2BGR)))
        print(
            f"  {f.stem:34s}  warp_err={m.warping_error:6.2f}  "
            f"con_l1={m.consecutive_l1:6.2f}  ratio={m.consistency_ratio:.3f}  "
            f"(n={m.n_pairs})"
        )

    # Compare against the lowest warping_error config (= best temporal consistency).
    best = min(rows, key=lambda r: r[1].warping_error)[0] if rows else None

    print("\n" + "=" * 78)
    print(
        f"{'config':34s}  {'warp_err':>9s}  {'con_l1':>7s}  {'ratio':>6s}  vs_best"
    )
    print("-" * 78)
    base_we = next((r[1].warping_error for r in rows if r[0] == best), 0.0) or 1.0
    for name, m in rows:
        rel = m.warping_error / base_we if base_we > 0 else 0.0
        marker = "  <-- best" if name == best else ""
        print(
            f"{name:34s}  {m.warping_error:9.3f}  {m.consecutive_l1:7.3f}  "
            f"{m.consistency_ratio:6.3f}  {rel:.2f}x{marker}"
        )

    if grid_panels:
        grid = make_grid(grid_panels)
        Path(args.grid_out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.grid_out, grid)
        print(f"\n[grid] saved {args.grid_out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
