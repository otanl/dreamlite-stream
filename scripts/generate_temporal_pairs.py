"""Generate frame-pair training data for temporal LLLite from a DAVIS sequence.

For each consecutive pair (frame_t, frame_{t+1}) in the input video:
  1. Run base DreamLite on each frame to get stylized targets target_t, target_{t+1}.
  2. Compute optical flow F: in_t -> in_{t+1} (Farneback, image-space).
  3. Warp target_t by F (cv2.remap with forward-flow / backward-warp convention).
  4. Save the training tuple:
       in_{t+1}         (RGB uint8)         input image for next-frame edit
       warped_target_t  (RGB uint8)         conditioning for LLLite
       target_{t+1}     (RGB uint8)         training target
       prompt                               stylization text

Output layout:
    <out_dir>/pairs/<seq>_<idx:05d>.npz   one file per training pair
    <out_dir>/manifest.jsonl              one JSON line per pair

Pair generation uses the MVP-1.5 fast runtime (compile + pipelined) so a
30-frame DAVIS clip costs ~5s after compile (one-time ~3 min).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402

from dreamlite_stream import EditWorker, SharedState  # noqa: E402
from dreamlite_stream import flow as flowlib  # noqa: E402
from dreamlite_stream.metrics import read_video_frames  # noqa: E402
from dreamlite_stream.output_blend import OutputBlender  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--inputs", nargs="+", required=True,
                   help="one or more DAVIS-converted mp4 files")
    p.add_argument("--prompt", required=True,
                   help="stylization prompt; same for all sequences")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=4,
                   help="use 4 for high-quality target generation, 2 for faster")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames_per_seq", type=int, default=None,
                   help="cap frames per sequence (e.g. 50 for fast pilot)")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument(
        "--blend_alpha", type=float, default=0.85,
        help="post-hoc output blend alpha for target generation. "
             "1.0 = raw base targets (flickery). 0.85 = distillation supervision (recommended)."
    )
    p.add_argument("--out_dir", default="data/temporal_pairs")
    return p.parse_args()


@torch.no_grad()
def stylize_sequence(
    worker: EditWorker, frames: List[np.ndarray], blend_alpha: float = 1.0,
) -> List[np.ndarray]:
    """Run worker per-frame; optionally apply post-hoc output blending.

    With blend_alpha=1.0, returns raw base-model outputs (flickery). With
    blend_alpha<1.0 (e.g. 0.85) the outputs are smoothed using OutputBlender —
    these become the "consistent supervision" used to distill the LLLite.
    """
    from PIL import Image
    blender = OutputBlender(alpha=blend_alpha) if blend_alpha < 1.0 else None
    out_frames = []
    for f in frames:
        out_pil, _ = worker.step(Image.fromarray(f))
        out_rgb = np.asarray(out_pil.convert("RGB"))
        if blender is not None:
            out_rgb = blender.apply(out_rgb, f)
        out_frames.append(out_rgb)
    return out_frames


def make_pairs(
    seq_name: str,
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
    out_dir: Path,
    prompt: str,
) -> int:
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
            xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            # Forward warp tgt_a (= time-t target) to time t+1 coords: MINUS flow.
            map_x = xs - flow[..., 0]
            map_y = ys - flow[..., 1]
            warped_tgt_a = cv2.remap(
                tgt_a, map_x, map_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            stem = f"{seq_name}_{t:05d}"
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
    # Wipe manifest if starting fresh.
    manifest = out_dir / "manifest.jsonl"
    if manifest.exists():
        manifest.unlink()

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

    total_pairs = 0
    for in_path in args.inputs:
        seq_name = Path(in_path).stem
        print(f"\n=== {seq_name} ===")
        frames = read_video_frames(in_path, size=args.size)
        if args.max_frames_per_seq:
            frames = frames[: args.max_frames_per_seq]
        print(f"  frames: {len(frames)}")
        # Stylize per-frame (no temporal coupling).
        targets = stylize_sequence(worker, frames, blend_alpha=args.blend_alpha)
        # Reset state so the next sequence starts clean.
        state.reset()
        n_pairs = make_pairs(seq_name, frames, targets, out_dir, args.prompt)
        total_pairs += n_pairs
        print(f"  pairs: {n_pairs}  cumulative: {total_pairs}")

    print(f"\n[done] {total_pairs} pairs written to {out_dir}")


if __name__ == "__main__":
    main()
