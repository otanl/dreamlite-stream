"""Render a side-by-side preview of generated temporal pairs.

Usage:
    python scripts/viz_temporal_pair.py --pairs_dir data/temporal_pairs/pairs \
        --n 6 --out preview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", required=True)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--out", default="out/temporal_pairs_preview.png")
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(Path(args.pairs_dir).glob("*.npz"))[: args.n]
    if not files:
        raise FileNotFoundError(args.pairs_dir)

    rows = []
    for f in files:
        d = np.load(f)
        in_rgb = d["input"]
        wprev_rgb = d["warped_target_prev"]
        tgt_rgb = d["target"]
        h, w, _ = in_rgb.shape
        labels = ["input (in_t+1)", "warped_target_t (cond)", "target (target_t+1)"]
        cells = [in_rgb, wprev_rgb, tgt_rgb]
        row = np.concatenate([cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for c in cells], axis=1)
        # caption bar
        cap = np.full((28, row.shape[1], 3), 32, dtype=np.uint8)
        for i, label in enumerate(labels):
            cv2.putText(
                cap, f"[{f.stem}] {label}" if i == 0 else label,
                (i * w + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (240, 240, 240), 1, cv2.LINE_AA,
            )
        rows.append(np.concatenate([cap, row], axis=0))

    grid = np.concatenate(rows, axis=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"saved {args.out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
