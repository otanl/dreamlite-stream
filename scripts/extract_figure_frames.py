"""Extract frames for paper figures:
  (2) Multi-prompt qualitative grid:  3 clips x (input + 4 styles) = 15 PNGs
  (3) Scene-cut visual:  2 N-configs x 8 frames (28-35) on blackswan_goat = 16 PNGs
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent

# === Multi-prompt grid sources ===
CLIPS = ["blackswan", "dance-twirl", "parkour"]
FRAME_IDX = 16
GRID_OUT = _ROOT / "figures_for_paper" / "qualitative_grid"

DAVIS = _ROOT / "assets" / "davis_mp4"
CHAMPION = _ROOT / "out" / "champion" / "champion"  # oil-painting champion
HELDOUT = _ROOT / "out" / "heldout_prompts_eval_v3"

# 4 styles + 1 input = 5 rows total in the grid
STYLES = [
    ("input", lambda seq: DAVIS / f"{seq}.mp4"),
    ("oil_champion", lambda seq: CHAMPION / f"{seq}.mp4"),
    ("comic_heldout", lambda seq: HELDOUT / f"comic_{seq}.mp4"),
    ("ukiyoe_heldout", lambda seq: HELDOUT / f"ukiyoe_{seq}.mp4"),
    ("vangogh_heldout", lambda seq: HELDOUT / f"vangogh_{seq}.mp4"),
]

# === Scene-cut sources ===
SCENECUT_CLIP = "blackswan_goat"
SCENECUT_OUT = _ROOT / "figures_for_paper" / "scenecut"
SCENECUT_FRAMES = list(range(28, 36))  # 28..35 inclusive (4 pre + 4 post; cut at 32)
SCENECUT_DIRS = {
    "N8": _ROOT / "out" / "scene_cut" / "N8" / f"{SCENECUT_CLIP}.mp4",
    "N1": _ROOT / "out" / "scene_cut" / "N1" / f"{SCENECUT_CLIP}.mp4",
}


def extract_frame(mp4_path: Path, frame_idx: int, out_png: Path, size: int = 256):
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
    # center-crop square if needed (inputs are 854x480; outputs are already 512x512)
    h, w, _ = bgr.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    bgr = bgr[y0 : y0 + s, x0 : x0 + s]
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), bgr)
    return True


def main():
    # Grid frames
    print("[1] qualitative_grid")
    for style, fn in STYLES:
        for clip in CLIPS:
            src = fn(clip)
            if not src.exists():
                print(f"  miss: {src}")
                continue
            out = GRID_OUT / f"{style}_{clip}.png"
            if extract_frame(src, FRAME_IDX, out):
                print(f"  ok   {style}/{clip} -> {out.name}")

    # Scene-cut frames
    print("\n[2] scene_cut")
    for nlabel, src in SCENECUT_DIRS.items():
        if not src.exists():
            print(f"  miss: {src}")
            continue
        for f in SCENECUT_FRAMES:
            out = SCENECUT_OUT / f"{nlabel}_f{f:02d}.png"
            if extract_frame(src, f, out, size=256):
                print(f"  ok   {nlabel} frame {f:02d} -> {out.name}")

    print(f"\nfigures saved under {_ROOT / 'figures_for_paper'}")


if __name__ == "__main__":
    main()
