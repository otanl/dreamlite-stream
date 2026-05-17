"""Generate distillation pairs for held-out video experiment.

Trains on 7 of the 10 DAVIS sequences we used originally, evaluates
on the remaining 3. The split is stratified by motion class so train
and eval span similar motion ranges.

Train (7): blackswan, libby, swing, camel, dance-twirl, goat, scooter-black
Eval  (3): bmx-trees, parkour, kite-surf

Same oil-painting prompt as v3 LLLite (single-prompt training).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite_stream import EditWorker, SharedState
from dreamlite_stream import flow as flowlib
from dreamlite_stream.metrics import read_video_frames
from dreamlite_stream.output_blend import OutputBlender  # noqa: F401
sys.path.insert(0, str(_ROOT / "scripts"))
from generate_temporal_pairs import stylize_sequence, make_pairs  # type: ignore  # noqa: E402
from dreamlite import DreamLiteMobilePipeline


TRAIN_SEQUENCES = [
    "blackswan", "libby", "swing", "camel", "dance-twirl",
    "goat", "scooter-black",
]
# Eval sequences (held-out from training): bmx-trees, parkour, kite-surf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--mp4_dir", default=str(_ROOT / "assets" / "davis_mp4"))
    p.add_argument("--sequences", nargs="+", default=TRAIN_SEQUENCES)
    p.add_argument("--out_dir", default=str(_ROOT / "data" / "temporal_pairs_v5_heldout_video"))
    p.add_argument("--max_frames_per_seq", type=int, default=50)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--blend_alpha", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--no_compile", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    if manifest.exists():
        manifest.unlink()

    inputs = [Path(args.mp4_dir) / f"{s}.mp4" for s in args.sequences]
    inputs = [p for p in inputs if p.exists()]
    print(f"[setup] {len(inputs)} train sequences, prompt: {args.prompt!r}")

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=args.steps, prompt=args.prompt,
    )
    worker = EditWorker(
        pipeline=pipeline, state=state, device=args.device,
        dtype=torch.bfloat16, init_mode="pure", seed=args.seed,
        compile=not args.no_compile, compile_mode="reduce-overhead",
    )

    total = 0
    for path in inputs:
        seq = path.stem
        frames = read_video_frames(str(path), size=args.size)
        if args.max_frames_per_seq:
            frames = frames[: args.max_frames_per_seq]
        print(f"\n[{seq}] stylizing {len(frames)} frames...")
        targets = stylize_sequence(worker, frames, blend_alpha=args.blend_alpha)
        state.reset()
        n = make_pairs(seq, frames, targets, out_dir, args.prompt)
        total += n
        print(f"  pairs: {n}  cumulative: {total}")

    print(f"\n[done] {total} pairs written to {out_dir}")


if __name__ == "__main__":
    main()
