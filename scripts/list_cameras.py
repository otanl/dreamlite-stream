"""Probe cv2.VideoCapture indices 0..N and report which ones work.

Usage:
    python scripts/list_cameras.py [--max_index 5] [--save_thumbs]

For each working index, prints (width, height, fps_property), the
brightness of the picture it delivers, and the backend cv2 chose. With
--save_thumbs, captures one frame per index and writes thumb_cam<i>.png
so you can tell which physical camera is which.

Sources commonly return black frames for the first fraction of a second
after opening, so the probe reads on for --warmup_s seconds and judges
(and saves) the last frame, not the first. An index that is still black
after that is delivering no picture: on Windows this is typically a
virtual camera (NDI Webcam, OBS Virtual Camera, a linked phone) whose
source application is not running.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2


def dshow_device_names() -> list[str]:
    """DirectShow video device names in enumeration order, via ffmpeg.

    Best effort: returns [] off Windows or when ffmpeg is not on PATH. The
    order matches the indices of the dshow backend; other backends may
    number devices differently or expose fewer of them.
    """
    if not sys.platform.startswith("win") or shutil.which("ffmpeg") is None:
        return []
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        )
    except Exception:
        return []
    return re.findall(r'"([^"]+)" \(video\)', res.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_index", type=int, default=5,
                   help="Probe indices 0..max_index inclusive")
    p.add_argument("--save_thumbs", action="store_true",
                   help="Save one frame per working camera as thumb_camN.png")
    p.add_argument("--backend", choices=["any", "dshow", "msmf", "v4l2"], default="any",
                   help="Force a cv2 capture backend (Windows: dshow or msmf)")
    p.add_argument("--warmup_s", type=float, default=1.5,
                   help="Seconds of frames to read before judging / saving a frame")
    args = p.parse_args()

    backend_map = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,    # Windows DirectShow
        "msmf":  cv2.CAP_MSMF,     # Windows Media Foundation
        "v4l2":  cv2.CAP_V4L2,     # Linux
    }
    backend = backend_map[args.backend]

    names = dshow_device_names()
    if names:
        print("DirectShow video devices, in order:")
        for i, name in enumerate(names):
            print(f"  {i}: {name}")
        print()

    print(f"Probing indices 0..{args.max_index} (backend={args.backend})\n")
    found = []
    lit = []
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
        # The first frames after opening are often black. Keep reading for a
        # moment and judge the last frame that arrived.
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.warmup_s:
            ok, nxt = cap.read()
            if not ok or nxt is None:
                break
            frame = nxt
        brightness = float(frame.mean())
        if brightness < 1.0:
            note = "  BLACK: no picture"
        elif brightness < 10.0:
            note = "  very dark"
        else:
            note = ""
            lit.append(i)
        print(f"  [{i}]  {w}x{h} @ {fps:.1f} fps   (frame shape: {frame.shape})  "
              f"brightness {brightness:.0f}/255{note}")
        if args.save_thumbs:
            thumb = Path(f"thumb_cam{i}.png")
            cv2.imwrite(str(thumb), frame)
            print(f"        saved -> {thumb}")
        found.append(i)
        cap.release()

    print(f"\nWorking indices: {found}")
    if found and not lit:
        print("\nNone of them is delivering a picture. Virtual cameras (NDI Webcam, OBS")
        print("Virtual Camera, a linked phone) stay black until their source application")
        print("is running and sending video.")
    if lit:
        print(f"Delivering a picture: {lit}")
        print(f"\nLaunch the demo with e.g.:")
        print(f"  python scripts/demo_camera.py --camera {lit[0]}")
    elif found:
        print(f"\nOnce a source is live, launch the demo with e.g.:")
        print(f"  python scripts/demo_camera.py --camera {found[0]}")


if __name__ == "__main__":
    main()
