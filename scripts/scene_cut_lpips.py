"""LPIPS between N=8 default and N=1 always-refresh oracle on scene-cut clips.

Scene-cut Appendix-B note: epsilon_w is indistinguishable between N=8 and N=1
because a stale-but-fixed cond_emb still produces a temporally consistent
output. LPIPS against the always-refresh oracle is a visual-fidelity probe
that can surface the lag.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dreamlite_stream.metrics import read_video_frames  # noqa: E402

SCENE_CUT_OUT = _ROOT / "out" / "scene_cut"
CLIPS = ["blackswan_goat", "kite_dance", "libby_camel"]
PRE_LEN = 32
POST_LEN = 32


def _lpips_pairwise(frames_a, frames_b, net):
    """Mean LPIPS (alex) between paired uint8 RGB frame lists."""
    if len(frames_a) == 0:
        return float("nan")
    n = min(len(frames_a), len(frames_b))
    ds = []
    for i in range(n):
        a = torch.from_numpy(frames_a[i]).permute(2, 0, 1).unsqueeze(0).float().cuda() / 127.5 - 1.0
        b = torch.from_numpy(frames_b[i]).permute(2, 0, 1).unsqueeze(0).float().cuda() / 127.5 - 1.0
        with torch.no_grad():
            d = net(a, b).item()
        ds.append(d)
    return float(np.mean(ds))


def main():
    import lpips
    net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()

    rows = []
    for clip in CLIPS:
        pN8 = SCENE_CUT_OUT / "N8" / f"{clip}.mp4"
        pN1 = SCENE_CUT_OUT / "N1" / f"{clip}.mp4"
        if not pN8.exists() or not pN1.exists():
            print(f"[skip] {clip}: missing mp4")
            continue
        f8 = read_video_frames(str(pN8))[:PRE_LEN + POST_LEN]
        f1 = read_video_frames(str(pN1))[:PRE_LEN + POST_LEN]
        n = min(len(f8), len(f1))
        f8 = f8[:n]
        f1 = f1[:n]
        if n < PRE_LEN + 16:
            print(f"[skip] {clip}: only {n} frames available")
            continue

        d_pre = _lpips_pairwise(f8[0:PRE_LEN], f1[0:PRE_LEN], net)
        d_post8 = _lpips_pairwise(f8[PRE_LEN:PRE_LEN + 8], f1[PRE_LEN:PRE_LEN + 8], net)
        d_post16 = _lpips_pairwise(f8[PRE_LEN:PRE_LEN + 16], f1[PRE_LEN:PRE_LEN + 16], net)

        print(f"  {clip:18s} pre={d_pre:.4f}  post8={d_post8:.4f}  post16={d_post16:.4f}")
        rows.append({
            "clip": clip,
            "lpips_pre_n8_vs_n1": d_pre,
            "lpips_post8_n8_vs_n1": d_post8,
            "lpips_post16_n8_vs_n1": d_post16,
        })

    if rows:
        pres = [r["lpips_pre_n8_vs_n1"] for r in rows]
        p8 = [r["lpips_post8_n8_vs_n1"] for r in rows]
        p16 = [r["lpips_post16_n8_vs_n1"] for r in rows]
        def fmt(v):
            return f"{mean(v):.4f}+-{stdev(v) if len(v)>1 else 0:.4f}"
        print("\n=== Aggregate (mean +- std over clips) ===")
        print(f"  pre   = {fmt(pres)}")
        print(f"  post8 = {fmt(p8)}")
        print(f"  post16= {fmt(p16)}")

    out_jsonl = SCENE_CUT_OUT / "lpips_n8_vs_n1.jsonl"
    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[saved] {out_jsonl}")


if __name__ == "__main__":
    main()
