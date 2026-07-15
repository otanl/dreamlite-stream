"""B3 vs O3 head-to-head comparison for the follow-up paper.

Runs `eval_lcm_lora.py` twice with consistent settings (the final
12-epoch checkpoint of each), then aggregates the per-sequence
metrics into a single comparison table.

Usage::

    python scripts/compare_b3_o3.py \
        --b3_weights runs/lcm_lora_B3/lcm_lora_step001644.pt \
        --o3_weights runs/lcm_lora_O3/lcm_lora_step001644.pt

This script does NOT include LLLite — both runs use the bare
LCM-LoRA + UNet so the comparison isolates the loss formulation.

Outputs:
- ``out/compare_b3_vs_o3/B3/results.jsonl``
- ``out/compare_b3_vs_o3/O3/results.jsonl``
- ``out/compare_b3_vs_o3/summary.md`` (markdown comparison table)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--b3_weights",
        default=str(_ROOT / "runs" / "lcm_lora_B3" / "lcm_lora_step001644.pt"),
        help="Path to the B3 (baseline MSE) final checkpoint.",
    )
    p.add_argument(
        "--o3_weights",
        default=str(_ROOT / "runs" / "lcm_lora_O3" / "lcm_lora_step001644.pt"),
        help="Path to the O3 (proposed loss) final checkpoint.",
    )
    p.add_argument("--out_dir", default=str(_ROOT / "out" / "compare_b3_vs_o3"))
    p.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for eval; 8 matches our DAVIS-10 eval protocol.",
    )
    p.add_argument(
        "--skip_eval", action="store_true",
        help="Skip the eval_lcm_lora.py runs and just regenerate the summary "
             "table from existing JSONLs.",
    )
    return p.parse_args()


def run_eval(weights: str, out_subdir: Path, batch_size: int) -> None:
    """Invoke eval_lcm_lora.py with the given weights and output dir."""
    out_subdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "eval_lcm_lora.py"),
        "--lcm_lora_weights", weights,
        "--out_dir", str(out_subdir),
        "--batch_size", str(batch_size),
    ]
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def load_results(jsonl_path: Path) -> list:
    """Load per-sequence rows from a results.jsonl produced by eval_lcm_lora."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found; eval likely failed")
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate(rows: list, keys: list[str]) -> dict:
    """Return {key: (mean, std)} for each numeric key in `keys`."""
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and r[k] is not None]
        if not vals:
            out[k] = (None, None)
            continue
        out[k] = (mean(vals), stdev(vals) if len(vals) >= 2 else 0.0)
    return out


def fmt(v, prec=2):
    if v is None:
        return "—"
    return f"{v:.{prec}f}"


def write_summary(out_dir: Path, b3_agg: dict, o3_agg: dict) -> None:
    """Write a markdown comparison summary."""
    keys = ["fps", "warping_error", "sobel", "hf_fft", "lpips_teacher"]
    labels = {
        "fps": "fps",
        "warping_error": "ε_w",
        "sobel": "Sobel",
        "hf_fft": "HF-FFT",
        "lpips_teacher": "LPIPS-to-teacher",
    }

    md = []
    md.append("# B3 vs O3 — Phase 2 head-to-head\n")
    md.append("LCM-LoRA distillation comparison on DreamLite-mobile, "
              "DAVIS-9, K=1, B=8, no LLLite.\n")
    md.append("- **B3**: vanilla MSE baseline (§V LCM-LoRA v3 equivalent).\n")
    md.append("- **O3**: MSE + LPIPS (β=0.1) + HF-FFT log-ratio (γ=0.05).\n\n")

    md.append("| Metric | B3 (baseline) | O3 (proposed) | Δ vs B3 |\n")
    md.append("|---|---:|---:|---:|\n")
    for k in keys:
        b_m, b_s = b3_agg.get(k, (None, None))
        o_m, o_s = o3_agg.get(k, (None, None))
        prec = 2 if k != "hf_fft" else 0
        b_str = f"{fmt(b_m, prec)} ± {fmt(b_s, prec)}" if b_m is not None else "—"
        o_str = f"{fmt(o_m, prec)} ± {fmt(o_s, prec)}" if o_m is not None else "—"
        if b_m is not None and o_m is not None and b_m != 0:
            delta_pct = (o_m - b_m) / abs(b_m) * 100.0
            delta_str = f"{delta_pct:+.1f}%"
        else:
            delta_str = "—"
        md.append(f"| **{labels[k]}** | {b_str} | {o_str} | {delta_str} |\n")

    md.append("\n## Smoothing-detection signature\n\n")
    md.append("Expected if O3 prevents the smoothing collapse the §V case study\n")
    md.append("documented:\n")
    md.append("- ε_w should *not* differ dramatically (both are 1-step distilled "
              "students; the gameable improvement is what B3 *exhibits*).\n")
    md.append("- **Sobel and HF-FFT should be substantially higher for O3**, "
              "ideally approaching the 4-step teacher reference (~4.74 / 1760 "
              "from the champion).\n")
    md.append("- LPIPS-to-teacher should be **lower** for O3 (closer to teacher "
              "perceptually).\n\n")
    md.append("If the actual numbers above show this pattern, the proposed loss "
              "is doing what the paper claims; if not, debugging required.\n")

    out_md = out_dir / "summary.md"
    out_md.write_text("".join(md))
    print(f"[saved] {out_md}", flush=True)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b3_out = out_dir / "B3"
    o3_out = out_dir / "O3"

    if not args.skip_eval:
        # Verify weight files exist before launching eval (cheap sanity check).
        for label, weights in (("B3", args.b3_weights), ("O3", args.o3_weights)):
            if not Path(weights).exists():
                raise FileNotFoundError(
                    f"{label} weights not found at {weights}; "
                    "check that training has completed."
                )
        run_eval(args.b3_weights, b3_out, args.batch_size)
        run_eval(args.o3_weights, o3_out, args.batch_size)

    b3_rows = load_results(b3_out / "results.jsonl")
    o3_rows = load_results(o3_out / "results.jsonl")

    keys = ["fps", "warping_error", "sobel", "hf_fft", "lpips_teacher"]
    b3_agg = aggregate(b3_rows, keys)
    o3_agg = aggregate(o3_rows, keys)

    write_summary(out_dir, b3_agg, o3_agg)

    # Also dump the per-sequence rows side-by-side for inspection.
    combined = []
    for b, o in zip(b3_rows, o3_rows):
        combined.append({"sequence": b.get("sequence", "?"),
                         "B3": {k: b.get(k) for k in keys},
                         "O3": {k: o.get(k) for k in keys}})
    (out_dir / "per_sequence.json").write_text(json.dumps(combined, indent=2))
    print(f"[saved] {out_dir / 'per_sequence.json'}", flush=True)


if __name__ == "__main__":
    main()
