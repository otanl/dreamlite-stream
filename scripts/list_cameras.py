"""Probe cv2.VideoCapture indices 0..N and report which ones work.

Usage:
    python scripts/list_cameras.py [--max_index 5] [--save_thumbs]

For each working index, prints (width, height, fps_property) and the
backend cv2 chose. With --save_thumbs, captures one frame per index
and writes thumb_cam<i>.png so you can tell which physical camera is
which.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_index", type=int, default=5,
                   help="Probe indices 0..max_index inclusive")
    p.add_argument("--save_thumbs", action="store_true",
                   help="Save one frame per working camera as thumb_camN.png")
    p.add_argument("--backend", choices=["any", "dshow", "msmf", "v4l2"], default="any",
                   help="Force a cv2 capture backend (Windows: dshow or msmf)")
    args = p.parse_args()

    backend_map = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,    # Windows DirectShow
        "msmf":  cv2.CAP_MSMF,     # Windows Media Foundation
        "v4l2":  cv2.CAP_V4L2,     # Linux
    }
    backend = backend_map[args.backend]

    print(f"Probing indices 0..{args.max_index} (backend={args.backend})\n")
    found = []
    for i in range(args.max_index + 1):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            print(f"  [{i}]  -- not available")
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"  [{i}]  opened but read() failed; treating as unusable")
            cap.release()
            continue
        print(f"  [{i}]  {w}x{h} @ {fps:.1f} fps   (frame shape: {frame.shape})")
        if args.save_thumbs:
            thumb = Path(f"thumb_cam{i}.png")
            cv2.imwrite(str(thumb), frame)
            print(f"        saved -> {thumb}")
        found.append(i)
        cap.release()

    print(f"\nWorking indices: {found}")
    if found:
        print(f"\nLaunch the demo with e.g.:")
        print(f"  python scripts/demo_camera.py --camera {found[0]}")


if __name__ == "__main__":
    main()
