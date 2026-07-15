"""Aggregate OpenVE-Bench judge scores into the per-category and overall
means used in the paper's Table comparable to arXiv:2512.07826 Table 2.

Reads ``out/openve/judge_<judge>.jsonl`` (per-pair scores), ``results.jsonl``
(per-pair fps), and emits both a plain-text table and a LaTeX row stub
compatible with SANA-Streaming's Table 1 layout:

    Method | Params | FPS | IC | C&D | VQ&S | mean

Sanity check (do this BEFORE believing our headline number): re-score the
released OpenVE-Edit model with the same judge and verify the reproduced
numbers fall within +-0.05 of the published (3.11, 2.72, 1.24) on Seed-1.6VL
(notes/openve_bench_plan.md §7).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parent.parent

_AXES = ("instruction_compliance", "consistency_detail", "visual_quality_stability")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--judge_file", required=True,
                   help="Per-pair judge JSONL (e.g. out/openve/judge_seed-1.6vl.jsonl)")
    p.add_argument("--results",
                   default=str(_ROOT / "out" / "openve" / "results.jsonl"),
                   help="Per-pair eval JSONL with fps numbers")
    p.add_argument("--method", default="Ours (DreamLite-stream LLLite v3)")
    p.add_argument("--params", default="0.39B U-Net + 2.13B MLLM TE")
    p.add_argument("--out_tex", default="")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def aggregate(judges: list[dict]) -> tuple[dict[str, dict], dict]:
    by_cat: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {a: [] for a in _AXES})
    overall: dict[str, list[float]] = {a: [] for a in _AXES}
    for rec in judges:
        cat = rec["category"]
        for a in _AXES:
            if a in rec and rec[a] is not None:
                by_cat[cat][a].append(float(rec[a]))
                overall[a].append(float(rec[a]))

    cat_means = {
        cat: {a: (mean(xs) if xs else float("nan")) for a, xs in axes.items()}
        for cat, axes in by_cat.items()
    }
    overall_mean = {a: (mean(xs) if xs else float("nan")) for a, xs in overall.items()}
    return cat_means, overall_mean


def print_table(cat_means: dict, overall_mean: dict, fps: float | None) -> None:
    print()
    print(f"{'Category':<22} {'IC':>6} {'C&D':>6} {'VQ&S':>6}")
    print("-" * 44)
    for cat, axes in sorted(cat_means.items()):
        ic = axes["instruction_compliance"]
        cd = axes["consistency_detail"]
        vq = axes["visual_quality_stability"]
        print(f"{cat:<22} {ic:>6.2f} {cd:>6.2f} {vq:>6.2f}")
    print("-" * 44)
    ic = overall_mean["instruction_compliance"]
    cd = overall_mean["consistency_detail"]
    vq = overall_mean["visual_quality_stability"]
    print(f"{'OVERALL':<22} {ic:>6.2f} {cd:>6.2f} {vq:>6.2f}")
    if fps is not None:
        print(f"\nMean fps over evaluated pairs: {fps:.2f}")


def emit_latex_row(method: str, params: str, fps: float | None,
                   overall_mean: dict) -> str:
    ic = overall_mean["instruction_compliance"]
    cd = overall_mean["consistency_detail"]
    vq = overall_mean["visual_quality_stability"]
    overall = mean([ic, cd, vq])
    fps_str = f"{fps:.1f}" if fps is not None else "--"
    return (
        f"{method} & {params} & {fps_str} & "
        f"{ic:.2f} & {cd:.2f} & {vq:.2f} & {overall:.2f} \\\\"
    )


def main() -> None:
    args = parse_args()
    judges = load_jsonl(Path(args.judge_file))
    if not judges:
        raise SystemExit(f"no judge records in {args.judge_file}")
    results = load_jsonl(Path(args.results))

    fps = None
    if results:
        scored = {r["pair_id"] for r in judges}
        fps_vals = [r["fps"] for r in results if r["pair_id"] in scored]
        if fps_vals:
            fps = mean(fps_vals)

    cat_means, overall_mean = aggregate(judges)
    print_table(cat_means, overall_mean, fps)

    latex = emit_latex_row(args.method, args.params, fps, overall_mean)
    print("\n--- LaTeX row stub ---")
    print(latex)

    if args.out_tex:
        Path(args.out_tex).write_text(latex + "\n", encoding="utf-8")
        print(f"\n[saved] {args.out_tex}")


if __name__ == "__main__":
    main()
