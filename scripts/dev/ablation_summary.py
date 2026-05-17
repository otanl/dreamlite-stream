"""Aggregate ablation_main.py results into paper-quality tables.

Outputs:
  - per-method summary across all sequences
  - per-motion-class breakdown (slow / medium / fast)
  - Pareto frontier in (fps, warp_err) space
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


# Manually classified DAVIS sequences by motion characteristics.
MOTION_CLASS = {
    "blackswan":   "slow",
    "libby":       "slow",
    "swing":       "slow",
    "camel":       "slow",
    "dance-twirl": "medium",
    "goat":        "medium",
    "kite-surf":   "medium",
    "scooter-black": "fast",
    "bmx-trees":   "fast",
    "parkour":     "fast",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="out/ablation/results.jsonl")
    p.add_argument("--out_csv", default="out/ablation/summary.csv")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    with open(args.results, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("(no rows)")
        return

    # Group by method
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    # Per-method aggregate
    methods_order = ["base_4st", "base_2st", "blend_2st", "lllite", "spec_t50", "spec_t200"]
    print("=" * 90)
    print(f"{'method':15s}  {'fps avg±std':>14s}  {'warp_err avg±std':>20s}  {'hit_rate':>8s}  N")
    print("-" * 90)
    summary_rows = []
    for m in methods_order:
        rs = by_method.get(m, [])
        if not rs:
            continue
        fps_vals = [r["fps"] for r in rs]
        we_vals = [r["warp_err"] for r in rs]
        hit_vals = [r["hit_rate"] for r in rs]
        fps_avg, fps_std = mean(fps_vals), stdev(fps_vals) if len(fps_vals) > 1 else 0.0
        we_avg, we_std = mean(we_vals), stdev(we_vals) if len(we_vals) > 1 else 0.0
        hit_avg = mean(hit_vals) * 100
        print(f"{m:15s}  {fps_avg:6.2f} ± {fps_std:5.2f}  "
              f"{we_avg:9.2f} ± {we_std:7.2f}  {hit_avg:7.1f}%  {len(rs)}")
        summary_rows.append({
            "method": m, "fps_avg": fps_avg, "fps_std": fps_std,
            "warp_err_avg": we_avg, "warp_err_std": we_std,
            "hit_rate_avg": hit_avg, "n": len(rs),
        })

    # Per-motion-class breakdown
    print("\n" + "=" * 90)
    print(f"{'method':15s}  {'class':>7s}  {'fps avg':>8s}  {'warp_err avg':>14s}  {'hit avg':>8s}  N")
    print("-" * 90)
    by_class = defaultdict(list)
    for r in rows:
        cls = MOTION_CLASS.get(r["sequence"], "?")
        by_class[(r["method"], cls)].append(r)
    for m in methods_order:
        for cls in ["slow", "medium", "fast"]:
            rs = by_class.get((m, cls), [])
            if not rs:
                continue
            fps_avg = mean(r["fps"] for r in rs)
            we_avg = mean(r["warp_err"] for r in rs)
            hit_avg = mean(r["hit_rate"] for r in rs) * 100
            print(f"{m:15s}  {cls:>7s}  {fps_avg:8.2f}  {we_avg:14.2f}  {hit_avg:7.1f}%  {len(rs)}")

    # Save summary CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[saved] {out_csv}")

    # Pareto: list per-row, sort by fps within method
    print("\nPareto-style (fps, warp_err) by sequence:")
    print(f"{'sequence':15s}  {'method':14s}  {'fps':>6s}  {'warp_err':>9s}")
    print("-" * 60)
    for r in sorted(rows, key=lambda r: (r["sequence"], -r["fps"])):
        print(
            f"{r['sequence']:15s}  {r['method']:14s}  "
            f"{r['fps']:6.2f}  {r['warp_err']:9.2f}"
        )


if __name__ == "__main__":
    main()
