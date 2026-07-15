"""Contextualise the warp-error gap: measure whether StreamV2V's lower warp
error is a smoothing-collapse artifact (softer output) rather than genuinely
better temporal fidelity.

For each of the 10 champion clips, compute per-frame:
  - Sobel mean-abs (absolute edge energy; lower = smoother)
  - HF-FFT log-ratio vs the DAVIS source (negative = lost HF; collapse)
for BOTH StreamV2V outputs and our champion outputs, at 512x512.

Run (main dreamlite env):
  python scripts/smoothing_collapse_vs_streamv2v.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
from smoothing_stress_test import sobel_mean_abs, hf_fft_log_ratio  # noqa

SV2V = _ROOT / "out" / "streamv2v_baseline"
OURS = _ROOT / "out" / "champion" / "champion"
SRC = _ROOT / "assets" / "davis_mp4_512sq"
CLIPS = ["blackswan", "bmx-trees", "camel", "dance-twirl", "goat",
         "kite-surf", "libby", "parkour", "scooter-black", "swing"]


def load(path: Path, n: int | None = None) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (512, 512):
            rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()
    t = torch.from_numpy(np.stack(frames)).float().permute(0, 3, 1, 2).contiguous() / 255
    return t if n is None else t[:n]


def main() -> None:
    rows = []
    for clip in CLIPS:
        src = load(SRC / f"{clip}.mp4")
        sv = load(SV2V / f"{clip}.mp4")
        ou = load(OURS / f"{clip}.mp4")
        n = min(len(src), len(sv), len(ou))
        src, sv, ou = src[:n], sv[:n], ou[:n]
        r = {
            "clip": clip, "n": n,
            "sobel_src": float(sobel_mean_abs(src).mean()),
            "sobel_sv2v": float(sobel_mean_abs(sv).mean()),
            "sobel_ours": float(sobel_mean_abs(ou).mean()),
            "hf_sv2v": float(hf_fft_log_ratio(sv, src).mean()),
            "hf_ours": float(hf_fft_log_ratio(ou, src).mean()),
        }
        rows.append(r)
        print(f"{clip:14s} Sobel src={r['sobel_src']:.3f} "
              f"SV2V={r['sobel_sv2v']:.3f} ours={r['sobel_ours']:.3f} | "
              f"HF-FFT SV2V={r['hf_sv2v']:+.3f} ours={r['hf_ours']:+.3f}")

    print("\n=== aggregate (10 clips) ===")
    for k in ["sobel_src", "sobel_sv2v", "sobel_ours", "hf_sv2v", "hf_ours"]:
        print(f"  {k:12s} {mean(r[k] for r in rows):+.4f}")
    print("\nInterpretation: if SV2V Sobel < ours AND HF-FFT(SV2V) more "
          "negative than ours, StreamV2V's lower warp error is (partly) a "
          "smoothing-collapse artifact, not superior temporal fidelity.")


if __name__ == "__main__":
    main()
