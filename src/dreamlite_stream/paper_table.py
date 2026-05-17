"""Paper-style results table generator.

Reads ablation_main.py output (results.jsonl) and produces:
  - main results table (Method × {avg FPS, avg warp_err, hit rate})
  - motion-class table (Method × {slow, medium, fast} for warp_err and FPS)
  - speedup vs quality scatter (CSV for plotting)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

# Sequence motion classification (manual; see DAVIS docs).
MOTION_CLASS = {
    "blackswan": "slow", "libby": "slow", "swing": "slow", "camel": "slow",
    "dance-twirl": "medium", "goat": "medium", "kite-surf": "medium",
    "scooter-black": "fast", "bmx-trees": "fast", "parkour": "fast",
}


METHOD_PRETTY = {
    "base_4st":   "Baseline (4-step, no compile)",
    "base_2st":   "Baseline (2-step)",
    "blend_2st":  "+ Output Blend (alpha=0.85)",
    "lllite":     "+ Temporal LLLite (m=1.0)",
    "spec_t50":   "+ Adaptive Spec (thresh=50px)",
    "spec_t200":  "+ Adaptive Spec (thresh=200px)",
}


def load_results(results_path: str) -> List[dict]:
    out = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def make_main_table(rows: List[dict]) -> str:
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    out = []
    out.append(f"{'Method':35s}  {'FPS':>10s}  {'Warp Error':>12s}  {'Hit Rate':>10s}  N")
    out.append("-" * 90)
    for m, name in METHOD_PRETTY.items():
        rs = by_method.get(m, [])
        if not rs:
            continue
        fps = [r["fps"] for r in rs]
        we = [r["warp_err"] for r in rs]
        hit = [r["hit_rate"] for r in rs]
        f_avg, f_std = mean(fps), (stdev(fps) if len(fps) > 1 else 0.0)
        w_avg, w_std = mean(we), (stdev(we) if len(we) > 1 else 0.0)
        h_avg = mean(hit) * 100
        out.append(
            f"{name:35s}  {f_avg:5.2f} ± {f_std:4.2f}  "
            f"{w_avg:6.2f} ± {w_std:4.2f}  {h_avg:9.1f}%  {len(rs)}"
        )
    return "\n".join(out)


def make_motion_table(rows: List[dict]) -> str:
    by_mc: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        cls = MOTION_CLASS.get(r["sequence"], "?")
        by_mc[(r["method"], cls)].append(r)
    out = []
    out.append(f"{'Method':35s}  {'slow FPS/WE':>14s}  {'med FPS/WE':>14s}  {'fast FPS/WE':>14s}")
    out.append("-" * 95)
    for m, name in METHOD_PRETTY.items():
        cells = []
        for cls in ("slow", "medium", "fast"):
            rs = by_mc.get((m, cls), [])
            if rs:
                f = mean(r["fps"] for r in rs)
                w = mean(r["warp_err"] for r in rs)
                cells.append(f"{f:5.2f}/{w:5.1f}")
            else:
                cells.append("    -      ")
        out.append(f"{name:35s}  {cells[0]:>14s}  {cells[1]:>14s}  {cells[2]:>14s}")
    return "\n".join(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="out/ablation/results.jsonl")
    args = ap.parse_args()

    rows = load_results(args.results)
    if not rows:
        print("(no rows)")
        return

    print("\n## Main Results (10 DAVIS sequences, 30 measured frames each)\n")
    print(make_main_table(rows))
    print("\n## Per-Motion-Class Breakdown (FPS / Warping Error)\n")
    print(make_motion_table(rows))


if __name__ == "__main__":
    main()
