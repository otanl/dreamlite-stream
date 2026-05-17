"""Generate multi-prompt distillation pairs for v4 LLLite training.

Single-process version that amortizes compile cost across all prompts.
Each pair carries its own prompt; per-prompt npz stems prevent collisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

from dreamlite_stream import EditWorker, SharedState
from dreamlite_stream import flow as flowlib
from dreamlite_stream.metrics import read_video_frames
from dreamlite_stream.output_blend import OutputBlender  # noqa: F401 (used via stylize_sequence)
sys.path.insert(0, str(_ROOT / "scripts"))
from generate_temporal_pairs import stylize_sequence  # type: ignore  # noqa: E402
from dreamlite import DreamLiteMobilePipeline


DEFAULT_PROMPTS = [
    "transfer this to oil painting style, vibrant colors",
    "transfer this to watercolor painting style, soft edges",
    "transfer this to pencil sketch style, fine line work",
    "transfer this to anime art style, clean cel shading",
    "transfer this to 3D render style, ray-traced lighting",
]

DEFAULT_SEQUENCES = [
    "blackswan", "libby", "swing", "camel", "dance-twirl",
    "goat", "scooter-black", "bmx-trees", "parkour", "kite-surf",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    p.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    p.add_argument("--mp4_dir", default=str(_ROOT / "assets" / "davis_mp4"))
    p.add_argument("--out_dir", default=str(_ROOT / "data" / "temporal_pairs_v4_multiprompt"))
    p.add_argument("--max_frames_per_seq", type=int, default=12)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--blend_alpha", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--no_compile", action="store_true")
    return p.parse_args()


def make_pairs_with_prefix(
    pidx: int, seq_name: str, inputs, targets, out_dir: Path, prompt: str
):
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    n = len(inputs)
    pair_count = 0
    with manifest_path.open("a", encoding="utf-8") as mf:
        for t in range(n - 1):
            in_a = inputs[t]
            in_b = inputs[t + 1]
            tgt_a = targets[t]
            tgt_b = targets[t + 1]
            in_a_g = flowlib.to_gray(in_a)
            in_b_g = flowlib.to_gray(in_b)
            flow = flowlib.farneback_flow(in_a_g, in_b_g)
            H, W = flow.shape[:2]
            xs, ys = np.meshgrid(
                np.arange(W, dtype=np.float32),
                np.arange(H, dtype=np.float32),
            )
            map_x = xs - flow[..., 0]
            map_y = ys - flow[..., 1]
            warped_tgt_a = cv2.remap(
                tgt_a, map_x, map_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            stem = f"p{pidx}_{seq_name}_{t:05d}"
            np.savez_compressed(
                pairs_dir / f"{stem}.npz",
                input=in_b,
                warped_target_prev=warped_tgt_a,
                target=tgt_b,
            )
            mf.write(json.dumps({
                "stem": stem,
                "seq": seq_name,
                "frame_idx": t + 1,
                "prompt": prompt,
                "size": int(H),
            }) + "\n")
            pair_count += 1
    return pair_count


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    if manifest.exists():
        manifest.unlink()

    inputs = [Path(args.mp4_dir) / f"{s}.mp4" for s in args.sequences]
    inputs = [p for p in inputs if p.exists()]
    print(f"[setup] {len(inputs)} sequences, {len(args.prompts)} prompts")

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    # Pre-load frames once per sequence (shared across prompts)
    seq_frames = {}
    for path in inputs:
        seq = path.stem
        frames = read_video_frames(str(path), size=args.size)
        if args.max_frames_per_seq:
            frames = frames[: args.max_frames_per_seq]
        seq_frames[seq] = frames
        print(f"  {seq}: {len(frames)} frames")

    total = 0
    for pidx, prompt in enumerate(args.prompts):
        print(f"\n========== prompt {pidx + 1}/{len(args.prompts)}: {prompt!r} ==========")
        state = SharedState(
            height=args.size, width=args.size,
            num_inference_steps=args.steps, prompt=prompt,
        )
        # Recreate worker so prompt embedding is rebuilt; reuse pipeline so
        # compile cache stays hot across prompts.
        worker = EditWorker(
            pipeline=pipeline, state=state, device=args.device,
            dtype=torch.bfloat16, init_mode="pure", seed=args.seed,
            compile=not args.no_compile, compile_mode="reduce-overhead",
        )

        for seq, frames in seq_frames.items():
            print(f"  [{seq}] stylizing {len(frames)} frames...")
            targets = stylize_sequence(worker, frames, blend_alpha=args.blend_alpha)
            state.reset()
            n = make_pairs_with_prefix(pidx, seq, frames, targets, out_dir, prompt)
            total += n
            print(f"    pairs: {n}  cumulative: {total}")

    print(f"\n[done] {total} pairs written to {out_dir}")


if __name__ == "__main__":
    main()
