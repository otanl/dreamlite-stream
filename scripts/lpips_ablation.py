"""Compute LPIPS perceptual distance between each ablation config and the
4-step baseline (teacher reference). Aggregates per-config across sequences.

Usage:
    lpips_ablation.py [--ablation_dir <path>] [--ref_config baseline_eager]

Outputs a printed table and writes lpips_summary.csv into ablation_dir.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from statistics import mean, stdev

import cv2
import lpips
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent


_DEFAULT_CONFIGS = [
    "baseline_eager", "compile_2st_blend",
    "base_b8_1st_pipe", "base_b16_1st_pipe",
    "lllite_v3_eager", "lllite_v3_compile",
    "champion_b8", "champion_b16", "champion_b16_nf4",
]
_DEFAULT_SEQUENCES = [
    "blackswan", "libby", "swing", "camel", "dance-twirl",
    "goat", "bmx-trees", "parkour", "kite-surf",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_dir", default=str(_ROOT / "out" / "comprehensive_ablation"))
    p.add_argument("--ref_config", default="baseline_eager")
    p.add_argument("--configs", nargs="+", default=_DEFAULT_CONFIGS)
    p.add_argument("--sequences", nargs="+", default=_DEFAULT_SEQUENCES)
    p.add_argument("--max_frames", type=int, default=24)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_frames(path: str, max_frames: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    frames = []
    for i in range(max_frames):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0) if frames else None


def to_tensor(frames: np.ndarray, device: str) -> torch.Tensor:
    # uint8 RGB (T, H, W, 3) -> float [-1, 1] (T, 3, H, W)
    t = torch.from_numpy(frames).to(device).float() / 127.5 - 1.0
    return t.permute(0, 3, 1, 2)


def main():
    args = parse_args()
    ablation_dir = Path(args.ablation_dir)
    print(f"[load] LPIPS AlexNet on {args.device}")
    loss_fn = lpips.LPIPS(net="alex", verbose=False).to(args.device).eval()

    rows = []
    for cfg in args.configs:
        seq_dists = []
        for seq in args.sequences:
            cfg_path = ablation_dir / cfg / f"{seq}.mp4"
            ref_path = ablation_dir / args.ref_config / f"{seq}.mp4"
            if not cfg_path.exists() or not ref_path.exists():
                continue
            cf = load_frames(str(cfg_path), args.max_frames)
            rf = load_frames(str(ref_path), args.max_frames)
            if cf is None or rf is None:
                continue
            n = min(len(cf), len(rf))
            cf, rf = cf[:n], rf[:n]
            ct = to_tensor(cf, args.device)
            rt = to_tensor(rf, args.device)
            with torch.no_grad():
                d = loss_fn(ct, rt).flatten().mean().item()
            seq_dists.append(d)
        if not seq_dists:
            continue
        m = mean(seq_dists)
        s = stdev(seq_dists) if len(seq_dists) >= 2 else 0.0
        rows.append((cfg, len(seq_dists), m, s))
        print(f"{cfg:<22} n={len(seq_dists):>2}  LPIPS_to_4step={m:.4f} ± {s:.4f}")

    out_csv = ablation_dir / "lpips_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "n", "lpips_to_4step_mean", "lpips_to_4step_std"])
        for cfg, n, m, s in rows:
            w.writerow([cfg, n, f"{m:.4f}", f"{s:.4f}"])
    print(f"\n[saved] {out_csv}")


if __name__ == "__main__":
    main()
