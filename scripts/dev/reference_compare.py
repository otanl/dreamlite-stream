"""Reference-based fidelity metric for ablation outputs.

Picks one method (default 'lllite') as the "reference quality" and reports
mean per-frame L1 between every other method's output and the reference.

This complements warping_error (which is gameable by spec via warp construction):
reference_l1 measures absolute fidelity to the full-quality model output.

Usage:
    python scripts/reference_compare.py \
        --bench_dir out/ablation/videos --reference lllite \
        --sequences blackswan dance-twirl ...
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dreamlite_stream.metrics import read_video_frames, reference_l1  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bench_dir", default=str(_ROOT / "out" / "ablation" / "videos"),
                   help="dir containing per-sequence subdirs of method mp4s")
    p.add_argument("--reference", default="lllite",
                   help="method name (without .mp4) treated as ground truth")
    p.add_argument("--sequences", nargs="+", default=None,
                   help="restrict to these sequence subdirs; default = all")
    p.add_argument("--max_frames", type=int, default=33)
    return p.parse_args()


def main():
    args = parse_args()
    bench_dir = Path(args.bench_dir)
    if not bench_dir.is_dir():
        raise FileNotFoundError(bench_dir)

    seqs = (
        args.sequences if args.sequences
        else sorted(p.name for p in bench_dir.iterdir() if p.is_dir())
    )

    by_method = defaultdict(list)  # method -> [l1 per seq]
    for seq in seqs:
        seq_dir = bench_dir / seq
        ref_path = seq_dir / f"{args.reference}.mp4"
        if not ref_path.exists():
            print(f"  skip {seq}: missing reference {ref_path}")
            continue
        ref_frames = read_video_frames(str(ref_path))[: args.max_frames]
        for mp4 in sorted(seq_dir.glob("*.mp4")):
            method = mp4.stem
            if method == args.reference:
                continue
            test_frames = read_video_frames(str(mp4))[: args.max_frames]
            l1 = reference_l1(test_frames, ref_frames)
            by_method[method].append((seq, l1))

    print(f"\n## Reference fidelity (lower = closer to {args.reference}-quality output)\n")
    print(f"{'method':18s}  {'L1 avg':>8s}  per-sequence:")
    print("-" * 70)
    for method in sorted(by_method.keys()):
        rows = by_method[method]
        avg_l1 = mean(l1 for _, l1 in rows)
        per_seq = "  ".join(f"{seq[:6]}={l1:.1f}" for seq, l1 in rows)
        print(f"{method:18s}  {avg_l1:8.2f}  {per_seq}")


if __name__ == "__main__":
    main()
