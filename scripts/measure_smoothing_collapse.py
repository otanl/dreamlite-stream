"""Measure smoothing-collapse readings on existing B3/O1/O2/O3 student outputs.

Re-derivation pivot (§3): instead of retraining DreamLite, walk the
already-generated outputs at ``out/compare_b3_vs_o3/{B3,O1,O2,O3}/``
and compute the two smoothing metrics on every frame, using the DAVIS source
video as the reference for HF-FFT log-ratio.

Output:
    - out/smoothing_rederivation/per_frame.csv: one row per (config, sequence, frame)
    - out/smoothing_rederivation/per_config.csv: aggregated per config
    - out/smoothing_rederivation/summary.md: human-readable table + interpretation

Runtime: ~5-10 minutes on CPU for 8 sequences x 4 configs x ~50 frames each.

Interpretation
==============
- Sobel mean-abs absolute value: lower for smoother outputs.
- HF-FFT log-ratio = log(student HF energy / DAVIS source HF energy):
  - 0 means the student preserved the HF content of the source.
  - Negative means the student lost HF content (smoothing collapse).
  - Positive means the student amplified HF (over-sharpening).

For §3 the load-bearing finding is:
    HF-FFT(B3) << HF-FFT(O3), with B3 well below zero (smoothing collapse)
    and O3 closer to zero (loss family prevents the collapse).
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

# Reuse the same metric definitions as scripts/smoothing_stress_test.py so the
# numbers reported here are directly comparable to the §3.4 stress-test plot.
sys.path.insert(0, str(Path(__file__).parent))
from smoothing_stress_test import sobel_mean_abs, hf_fft_log_ratio  # noqa: E402


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_mp4_as_frames(path: Path, max_frames: int = 64,
                       resolution: int = 512) -> torch.Tensor:
    """Decode an MP4 and return (N, 3, R, R) frames in [0, 1] on CPU.

    Frames are center-cropped to square then resized to `resolution`.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    out = []
    for f in frames:
        h, w = f.shape[:2]
        s = min(h, w)
        top, left = (h - s) // 2, (w - s) // 2
        f = f[top:top + s, left:left + s]
        f = cv2.resize(f, (resolution, resolution), interpolation=cv2.INTER_AREA)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        out.append(f)
    arr = np.stack(out, axis=0).astype(np.float32) / 255.0
    # (N, H, W, 3) -> (N, 3, H, W)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def find_davis_source(sequence_name: str, davis_root: Path,
                      max_frames: int = 64, resolution: int = 512
                      ) -> torch.Tensor | None:
    """Load DAVIS source frames for one sequence, matched to the MP4 stem.

    Returns (N, 3, R, R) tensor in [0, 1], or None if the DAVIS folder
    for that sequence is missing.
    """
    stem = Path(sequence_name).stem
    seq_dir = davis_root / stem
    if not seq_dir.is_dir():
        return None
    jpgs = sorted(seq_dir.glob("*.jpg"))[:max_frames]
    if not jpgs:
        return None
    out = []
    for j in jpgs:
        f = cv2.imread(str(j))
        if f is None:
            continue
        h, w = f.shape[:2]
        s = min(h, w)
        top, left = (h - s) // 2, (w - s) // 2
        f = f[top:top + s, left:left + s]
        f = cv2.resize(f, (resolution, resolution), interpolation=cv2.INTER_AREA)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        out.append(f)
    if not out:
        return None
    arr = np.stack(out, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


@dataclass
class FrameRow:
    config: str
    sequence: str
    frame_idx: int
    sobel_mean_abs: float
    hf_log_ratio: float


def evaluate_config(config_dir: Path, davis_root: Path, device: torch.device,
                    max_frames: int = 64, resolution: int = 512
                    ) -> list[FrameRow]:
    rows: list[FrameRow] = []
    mp4s = sorted(config_dir.glob("*.mp4"))
    if not mp4s:
        print(f"  (no mp4s in {config_dir})")
        return rows
    for mp4 in mp4s:
        seq = mp4.stem
        try:
            student = load_mp4_as_frames(mp4, max_frames, resolution).to(device)
        except RuntimeError as e:
            print(f"  skip {mp4.name}: {e}")
            continue
        source = find_davis_source(seq, davis_root, student.size(0), resolution)
        if source is None:
            print(f"  skip {seq}: no DAVIS source found at {davis_root / seq}")
            continue
        source = source.to(device)
        # Align lengths.
        n = min(student.size(0), source.size(0))
        student = student[:n]
        source = source[:n]
        sobel = sobel_mean_abs(student)         # (n,)
        hfr = hf_fft_log_ratio(student, source) # (n,)
        for i in range(n):
            rows.append(FrameRow(
                config=config_dir.name,
                sequence=seq,
                frame_idx=i,
                sobel_mean_abs=float(sobel[i].item()),
                hf_log_ratio=float(hfr[i].item()),
            ))
        print(f"  {mp4.name}: n={n}  Sobel mean={sobel.mean():.4f}  HFlog mean={hfr.mean():+.4f}")
    return rows


def write_per_frame_csv(path: Path, rows: list[FrameRow]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "sequence", "frame_idx",
                    "sobel_mean_abs", "hf_log_ratio"])
        for r in rows:
            w.writerow([r.config, r.sequence, r.frame_idx,
                        r.sobel_mean_abs, r.hf_log_ratio])


def aggregate_per_config(rows: list[FrameRow]) -> dict[str, dict[str, float]]:
    agg: dict[str, dict[str, float]] = {}
    by_config: dict[str, list[FrameRow]] = {}
    for r in rows:
        by_config.setdefault(r.config, []).append(r)
    for cfg, cfg_rows in by_config.items():
        sobel = np.array([r.sobel_mean_abs for r in cfg_rows])
        hf = np.array([r.hf_log_ratio for r in cfg_rows])
        agg[cfg] = {
            "n_frames": float(len(cfg_rows)),
            "sobel_mean": float(sobel.mean()),
            "sobel_std": float(sobel.std()),
            "hf_log_ratio_mean": float(hf.mean()),
            "hf_log_ratio_std": float(hf.std()),
        }
    return agg


def write_per_config_csv(path: Path, agg: dict[str, dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "n_frames", "sobel_mean", "sobel_std",
                    "hf_log_ratio_mean", "hf_log_ratio_std"])
        for cfg, stats in sorted(agg.items()):
            w.writerow([cfg, int(stats["n_frames"]),
                        stats["sobel_mean"], stats["sobel_std"],
                        stats["hf_log_ratio_mean"], stats["hf_log_ratio_std"]])


def write_summary(path: Path, agg: dict[str, dict[str, float]]) -> None:
    lines = [
        "# Smoothing-collapse re-derivation summary",
        "",
        "Generated by `scripts/measure_smoothing_collapse.py`.",
        "DAVIS source is the HF-FFT reference; log-ratio = log(student / source).",
        "",
        "| Config | n_frames | Sobel mean +/- std | HF-FFT log-ratio mean +/- std |",
        "|---|---|---|---|",
    ]
    for cfg in sorted(agg.keys()):
        s = agg[cfg]
        lines.append(
            f"| {cfg} | {int(s['n_frames'])} | "
            f"{s['sobel_mean']:.4f} +/- {s['sobel_std']:.4f} | "
            f"{s['hf_log_ratio_mean']:+.4f} +/- {s['hf_log_ratio_std']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Sobel decreases monotonically with smoothing; expected ordering",
        "  B3 < O1 / O2 < O3 if our loss family prevents smoothing collapse.",
        "- HF-FFT log-ratio close to 0 means the student preserved the source",
        "  HF energy; strongly negative means smoothing collapse.",
        "",
        "These numbers go directly into §3.4 (smoothing-collapse metric)",
        "and §5 (per-substrate ablation), DreamLite primary substrate column.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("out/compare_b3_vs_o3"))
    ap.add_argument("--davis-root", type=Path,
                    default=Path("assets/davis/DAVIS/JPEGImages/480p"))
    ap.add_argument("--configs", nargs="+",
                    default=["B3", "O1", "O2", "O3"])
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("out/smoothing_rederivation"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"device={device}, root={args.root}, configs={args.configs}")

    all_rows: list[FrameRow] = []
    for cfg in args.configs:
        cfg_dir = args.root / cfg
        if not cfg_dir.is_dir():
            print(f"missing config dir: {cfg_dir}")
            continue
        print(f"\n=== {cfg} ===")
        rows = evaluate_config(cfg_dir, args.davis_root, device,
                               args.max_frames, args.resolution)
        all_rows.extend(rows)

    if not all_rows:
        print("no rows; aborting")
        return 1

    write_per_frame_csv(args.out_dir / "per_frame.csv", all_rows)
    agg = aggregate_per_config(all_rows)
    write_per_config_csv(args.out_dir / "per_config.csv", agg)
    write_summary(args.out_dir / "summary.md", agg)

    print("\n=== per-config summary ===")
    for cfg in sorted(agg.keys()):
        s = agg[cfg]
        print(f"  {cfg}: Sobel={s['sobel_mean']:.4f}+/-{s['sobel_std']:.4f}  "
              f"HFlog={s['hf_log_ratio_mean']:+.4f}+/-{s['hf_log_ratio_std']:.4f}  "
              f"(n={int(s['n_frames'])})")

    # Quick directional check: does the expected ordering B3 < ... < O3 hold?
    if {"B3", "O3"}.issubset(agg.keys()):
        b3, o3 = agg["B3"], agg["O3"]
        print("\n=== §3 directional check ===")
        if b3["sobel_mean"] < o3["sobel_mean"]:
            print("  Sobel(B3) < Sobel(O3): O3 sharper than B3.  [OK]")
        else:
            print("  Sobel(B3) >= Sobel(O3): unexpected -- smoothing-collapse "
                  "did NOT manifest cleanly on B3.  Investigate.")
        if b3["hf_log_ratio_mean"] < o3["hf_log_ratio_mean"]:
            print("  HFlog(B3) < HFlog(O3): O3 closer to source HF.  [OK]")
        else:
            print("  HFlog(B3) >= HFlog(O3): unexpected.  Investigate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
