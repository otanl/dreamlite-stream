"""Extract frames for the B3 vs O3 qualitative side-by-side figure.

Top-3 sequences by Sobel Δ from the 2026-05-28 eval:
    swing    (+23.2%)
    parkour  (+23.1%)
    libby    (+21.4%)

Layout: 3 rows (Input / B3 / O3) x 3 cols (the 3 clips above), single frame each.
"""
from __future__ import annotations

from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent

CLIPS = ["swing", "parkour", "libby"]
FRAME_IDX = 16  # mid-clip — captures motion well

DAVIS = _ROOT / "assets" / "davis_mp4"
ROOT_OUT = _ROOT / "out" / "compare_b3_vs_o3_lllite"
CHAMPION = _ROOT / "out" / "champion" / "champion"

OUT_DIR = _ROOT / "figures_for_paper" / "b3_o3_grid"

ROWS = [
    ("input",    lambda seq: DAVIS / f"{seq}.mp4"),
    ("champion", lambda seq: CHAMPION / f"{seq}.mp4"),
    ("B3",       lambda seq: ROOT_OUT / "B3" / f"{seq}.mp4"),
    ("O1",       lambda seq: ROOT_OUT / "O1" / f"{seq}.mp4"),
    ("O2",       lambda seq: ROOT_OUT / "O2" / f"{seq}.mp4"),
    ("O3",       lambda seq: ROOT_OUT / "O3" / f"{seq}.mp4"),
]


def extract_frame(mp4_path: Path, frame_idx: int, out_png: Path, size: int = 320) -> bool:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"  [fail] cannot open {mp4_path}")
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        print(f"  [fail] frame {frame_idx} not readable in {mp4_path.name}")
        return False
    h, w, _ = bgr.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    bgr = bgr[y0 : y0 + s, x0 : x0 + s]
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), bgr)
    return True


def main():
    print(f"[extract] -> {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_total = 0
    for row_name, path_fn in ROWS:
        for clip in CLIPS:
            n_total += 1
            src = path_fn(clip)
            dst = OUT_DIR / f"{row_name}_{clip}.png"
            if extract_frame(src, FRAME_IDX, dst):
                print(f"  [ok] {row_name}/{clip} -> {dst.name}")
                n_ok += 1
    print(f"[done] {n_ok}/{n_total} frames extracted")


if __name__ == "__main__":
    main()
