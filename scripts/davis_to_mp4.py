"""Convert DAVIS-2017 JPEG sequences to mp4 for use in our pipeline.

DAVIS layout:
    DAVIS/JPEGImages/480p/<sequence>/00000.jpg, 00001.jpg, ...

Usage:
    python scripts/davis_to_mp4.py \
        --davis_root assets/davis/DAVIS \
        --sequences blackswan dance-twirl scooter-black \
        --out_dir assets/davis_mp4 --fps 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--davis_root", required=True,
                   help="path to extracted DAVIS root (containing JPEGImages/480p)")
    p.add_argument("--sequences", nargs="+", required=True,
                   help="sequence names under JPEGImages/480p")
    p.add_argument("--out_dir", default="assets/davis_mp4")
    p.add_argument("--fps", type=int, default=24)
    return p.parse_args()


def convert_one(seq_dir: Path, out_path: Path, fps: int) -> int:
    jpegs = sorted(seq_dir.glob("*.jpg"))
    if not jpegs:
        return 0
    first = cv2.imread(str(jpegs[0]))
    h, w, _ = first.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer: {out_path}")
    try:
        for j in jpegs:
            img = cv2.imread(str(j))
            if img is None:
                continue
            writer.write(img)
    finally:
        writer.release()
    return len(jpegs)


def main():
    args = parse_args()
    root = Path(args.davis_root) / "JPEGImages" / "480p"
    if not root.is_dir():
        raise FileNotFoundError(f"not found: {root}  (expected DAVIS/JPEGImages/480p layout)")
    out_dir = Path(args.out_dir)
    for seq in args.sequences:
        seq_dir = root / seq
        if not seq_dir.is_dir():
            print(f"  skip {seq}: missing")
            continue
        out_path = out_dir / f"{seq}.mp4"
        n = convert_one(seq_dir, out_path, args.fps)
        print(f"  {seq}: {n} frames -> {out_path}")


if __name__ == "__main__":
    main()
