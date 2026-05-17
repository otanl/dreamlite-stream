"""RAFT cross-check across multiple ordering configurations.

Round-6 reviewer note: the original RAFT cross-check (raft_warp_error.py) only
covered the champion configuration, but the body claim is about ordering
between configurations. To verify ordering preservation, we re-evaluate three
representative rows of Table 4:

  base_b16_1st_pipe   no-LLLite, K=1, B=16, pipelined  (Farneback eps_w 22.97)
  champion_b16        LLLite v3 + down_blocks + refresh, K=1, B=16  (18.34)
  lllite_v3_compile   LLLite v3 all-108 hooks, K=1, B=8, compile     (13.00)

If RAFT preserves the ordering 22.97 > 18.34 > 13.00, the body claim holds.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dreamlite_stream.metrics import read_video_frames  # noqa: E402

DAVIS_DIR = _ROOT / "assets" / "davis_mp4"
ABLATION_DIR = _ROOT / "out" / "comprehensive_ablation"
ABLATION_JSONL = ABLATION_DIR / "results.jsonl"

# Per-config: (display name in the body, sub-directory)
CONFIGS = [
    ("no-LLLite K=1 B=16 (pipelined)", "base_b16_1st_pipe"),
    ("Champion (LLLite v3 + down_blocks + refresh, B=16)", "champion_b16"),
    ("All-108-hook LLLite (compile, B=8)", "lllite_v3_compile"),
]
SEQS = [
    "blackswan", "libby", "swing", "camel", "dance-twirl",
    "goat", "scooter-black", "bmx-trees", "parkour", "kite-surf",
]


def _load_raft():
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights, progress=False).eval().cuda()
    transforms = weights.transforms()
    return model, transforms


def _raft_flow(model, transforms, prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> np.ndarray:
    a = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    b = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    a, b = transforms(a, b)
    a, b = a.cuda(), b.cuda()
    with torch.no_grad():
        flows = model(a, b)
    return flows[-1][0].permute(1, 2, 0).cpu().numpy().astype(np.float32)


def _warp_with_flow(img, flow):
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xs + flow[..., 0]).astype(np.float32)
    map_y = (ys + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def compute_raft_we(model, transforms, input_frames, output_frames):
    n = min(len(input_frames), len(output_frames))
    if n < 2:
        return 0.0
    we_sum = 0.0
    cnt = 0
    for t in range(n - 1):
        flow = _raft_flow(model, transforms, input_frames[t], input_frames[t + 1])
        out_a = output_frames[t].astype(np.float32)
        out_b = output_frames[t + 1].astype(np.float32)
        warped_b = _warp_with_flow(out_b, flow)
        we_sum += float(np.mean(np.abs(out_a - warped_b)))
        cnt += 1
    return we_sum / max(cnt, 1)


def main():
    # Index Farneback baseline per (config, seq)
    farneback = defaultdict(dict)
    with open(ABLATION_JSONL) as f:
        for line in f:
            r = json.loads(line)
            farneback[r["config"]][r["sequence"]] = r["warp_err"]

    model, transforms = _load_raft()

    all_rows = []
    for cfg_name, cfg_dir in CONFIGS:
        cfg_path = ABLATION_DIR / cfg_dir
        if not cfg_path.exists():
            print(f"[skip] config dir not found: {cfg_path}")
            continue
        print(f"\n=== {cfg_name} ({cfg_dir}) ===")
        rows = []
        for seq in SEQS:
            in_path = DAVIS_DIR / f"{seq}.mp4"
            out_path = cfg_path / f"{seq}.mp4"
            if not in_path.exists() or not out_path.exists():
                continue
            in_frames = read_video_frames(str(in_path), size=512)
            out_frames = read_video_frames(str(out_path), size=512)
            # Defensive: some no-LLLite B=16 runs have fewer frames (warmup
            # wall ate the tail). Match to the output count.
            n = min(len(in_frames), len(out_frames))
            in_frames, out_frames = in_frames[:n], out_frames[:n]
            we_raft = compute_raft_we(model, transforms, in_frames, out_frames)
            we_farneback = farneback[cfg_dir].get(seq, float("nan"))
            rel = (we_raft - we_farneback) / we_farneback * 100.0 if we_farneback else float("nan")
            print(f"  {seq:18s} RAFT={we_raft:6.2f}  Farneback={we_farneback:6.2f}  rel={rel:+.1f}%")
            rows.append({
                "config": cfg_dir, "sequence": seq,
                "warp_err_raft": we_raft,
                "warp_err_farneback": we_farneback,
                "relative_pct": rel,
            })
        all_rows.extend(rows)
        paired = [r for r in rows if not np.isnan(r["relative_pct"])]
        if paired:
            rafts = [r["warp_err_raft"] for r in paired]
            farnes = [r["warp_err_farneback"] for r in paired]
            rels = [r["relative_pct"] for r in paired]
            stdraft = stdev(rafts) if len(rafts) > 1 else 0
            stdfarn = stdev(farnes) if len(farnes) > 1 else 0
            stdrel = stdev(rels) if len(rels) > 1 else 0
            print(f"  mean RAFT      = {mean(rafts):6.2f} +- {stdraft:.2f}")
            print(f"  mean Farneback = {mean(farnes):6.2f} +- {stdfarn:.2f}")
            print(f"  mean relative  = {mean(rels):+5.1f}% +- {stdrel:.1f}%")

    out_jsonl = _ROOT / "out" / "raft_vs_farneback_ordering.jsonl"
    with open(out_jsonl, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[saved] {out_jsonl}")


if __name__ == "__main__":
    main()
