"""RAFT-based warping error cross-check.

We use Farneback for warping_error in the main paper (cv2.calcOpticalFlowFarneback)
because it has no learned dependency on the training data of the eval metric. To
verify that the flicker-vs-content trade-offs we report are not specific to
Farneback, this script re-computes warping_error using RAFT on the same champion
outputs and writes a side-by-side jsonl.

  warping_error_RAFT[s] := mean_t | out[s][t] - warp(out[s][t+1], RAFT(in[s][t], in[s][t+1])) |

against pixel range 0-255.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dreamlite_stream.metrics import read_video_frames  # noqa: E402

DAVIS_DIR = _ROOT / "assets" / "davis_mp4"
CHAMPION_DIR = _ROOT / "out" / "champion" / "champion"
RESULTS_JSONL = _ROOT / "out" / "champion" / "results.jsonl"
SEQS = [
    "blackswan", "libby", "swing", "camel", "dance-twirl",
    "goat", "scooter-black", "bmx-trees", "parkour", "kite-surf",
]


def _load_raft():
    """Load RAFT-Large from torchvision (no internet needed if cached)."""
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights, progress=False).eval().cuda()
    transforms = weights.transforms()
    return model, transforms


def _raft_flow(model, transforms, prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> np.ndarray:
    """Compute RAFT flow from prev->curr. Returns HxWx2 float32 numpy in pixel units."""
    a = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    b = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    a, b = transforms(a, b)
    a, b = a.cuda(), b.cuda()
    with torch.no_grad():
        flows = model(a, b)
    flow = flows[-1][0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return flow


def _warp_with_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xs + flow[..., 0]).astype(np.float32)
    map_y = (ys + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def compute_raft_warp_err(model, transforms, input_frames, output_frames) -> float:
    if len(input_frames) != len(output_frames):
        n = min(len(input_frames), len(output_frames))
        input_frames = input_frames[:n]
        output_frames = output_frames[:n]
    if len(input_frames) < 2:
        return 0.0
    we_sum = 0.0
    n = 0
    for t in range(len(input_frames) - 1):
        flow = _raft_flow(model, transforms, input_frames[t], input_frames[t + 1])
        out_a = output_frames[t].astype(np.float32)
        out_b = output_frames[t + 1].astype(np.float32)
        warped_b = _warp_with_flow(out_b, flow)
        we_sum += float(np.mean(np.abs(out_a - warped_b)))
        n += 1
    return we_sum / max(n, 1)


def main():
    farneback = {}
    with open(RESULTS_JSONL) as f:
        for line in f:
            r = json.loads(line)
            farneback[r["sequence"]] = r["warp_err"]

    model, transforms = _load_raft()
    rows = []
    for seq in SEQS:
        in_path = DAVIS_DIR / f"{seq}.mp4"
        out_path = CHAMPION_DIR / f"{seq}.mp4"
        if not in_path.exists() or not out_path.exists():
            print(f"  skip {seq}: missing input or output")
            continue
        in_frames = read_video_frames(str(in_path), size=512)
        out_frames = read_video_frames(str(out_path), size=512)
        we_raft = compute_raft_warp_err(model, transforms, in_frames, out_frames)
        we_farneback = farneback.get(seq, float("nan"))
        rel = (we_raft - we_farneback) / max(we_farneback, 1e-6) * 100.0
        print(f"  {seq:18s}  RAFT={we_raft:6.2f}  Farneback={we_farneback:6.2f}  rel={rel:+.1f}%")
        rows.append({
            "sequence": seq, "warp_err_raft": we_raft,
            "warp_err_farneback": we_farneback, "relative_pct": rel,
        })

    print()
    if rows:
        paired = [r for r in rows
                  if not (np.isnan(r["warp_err_farneback"]) or np.isnan(r["relative_pct"]))]
        rafts = [r["warp_err_raft"] for r in paired]
        farnes = [r["warp_err_farneback"] for r in paired]
        rels = [r["relative_pct"] for r in paired]
        print(f"  paired N = {len(paired)} (skipped {len(rows)-len(paired)} clip(s) lacking Farneback baseline)")
        if len(paired) > 1:
            print(f"  mean RAFT       = {mean(rafts):6.2f} +- {stdev(rafts):.2f}")
            print(f"  mean Farneback  = {mean(farnes):6.2f} +- {stdev(farnes):.2f}")
            print(f"  mean relative   = {mean(rels):+5.1f}% +- {stdev(rels):.1f}%")

    out_jsonl = _ROOT / "out" / "raft_vs_farneback_champion.jsonl"
    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[saved] {out_jsonl}")


if __name__ == "__main__":
    main()
