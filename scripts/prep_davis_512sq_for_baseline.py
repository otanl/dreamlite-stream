"""Pre-process the 10 champion DAVIS mp4s into 512x512 square clips that are
PIXEL-IDENTICAL to what champion_eval.py feeds our own pipeline, so an
external baseline (StreamV2V) consumes exactly the same inputs.

Replicates dreamlite_stream.runtime.iter_video_frames preprocessing:
  center-square-crop -> cv2.INTER_AREA resize to (size,size), cap at max_frames.
Writes with the same mp4v VideoWriter settings as runtime.VideoWriter.

Run (env with cv2):
  python scripts/prep_davis_512sq_for_baseline.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
CLIPS = ["blackswan", "bmx-trees", "camel", "dance-twirl", "goat",
         "kite-surf", "libby", "parkour", "scooter-black", "swing"]


def center_square_crop(rgb: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return rgb[y0:y0 + s, x0:x0 + s]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=str(_ROOT / "assets" / "davis_mp4"))
    ap.add_argument("--out_dir", default=str(_ROOT / "assets" / "davis_mp4_512sq"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max_frames", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for clip in CLIPS:
        in_path = Path(args.in_dir) / f"{clip}.mp4"
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            print(f"SKIP {clip}: cannot open {in_path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        out_path = out_dir / f"{clip}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps,
                                 (args.size, args.size))
        n = 0
        while n < args.max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = center_square_crop(rgb)
            rgb = cv2.resize(rgb, (args.size, args.size),
                             interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            n += 1
        writer.release()
        cap.release()
        print(f"{clip}: {n} frames -> {out_path} (fps={fps:.2f})")


if __name__ == "__main__":
    main()
