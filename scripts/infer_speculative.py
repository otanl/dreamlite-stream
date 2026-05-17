"""Inference with the adaptive speculative worker (optionally LLLite-augmented).

Wraps an EditWorker (the "miss path") with a SpeculativeEditWorker that
decides per-frame whether to skip full denoise (hit) or fall through (miss).

For meaningful hit rates the inner worker should produce temporally
consistent outputs — pass --lllite_weights to enable the trained adapter.

Usage:
    python scripts/infer_speculative.py \
        --input  assets/davis_mp4/dance-twirl.mp4 \
        --output out/speculative.mp4 \
        --prompt "transfer this to oil painting style, vibrant colors" \
        --lllite_weights runs/temporal_lllite_v1/temporal_lllite_step000420.safetensors \
        --flow_thresh 20 --max_consec 4
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import List

import torch
import torch._dynamo
from safetensors.torch import load_file

torch._dynamo.config.cache_size_limit = 64

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
_LLLITE = _ROOT.parent / "dreamlite-lllite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))
sys.path.insert(0, str(_LLLITE / "src"))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402
from dreamlite_lllite import apply_lllite  # noqa: E402

from dreamlite_stream import (  # noqa: E402
    EditWorker, SharedState, SpeculativeEditWorker,
)
from dreamlite_stream.runtime import VideoWriter, iter_video_frames  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="out/speculative.mp4")
    p.add_argument("--prompt", required=True)
    p.add_argument("--lllite_weights", default=None)
    p.add_argument("--lllite_multiplier", type=float, default=1.0)
    p.add_argument("--cond_emb_dim", type=int, default=32)
    p.add_argument("--mlp_dim", type=int, default=64)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--flow_thresh", type=float, default=20.0,
                   help="px; max flow magnitude tolerated before forcing a miss")
    p.add_argument("--max_consec", type=int, default=4,
                   help="bound consecutive hits to limit drift")
    p.add_argument("--compile", action="store_true",
                   help="compile inner worker (currently breaks under LLLite)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    controller = None
    if args.lllite_weights:
        vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
        latent_hw = args.size // vae_downsample
        controller = apply_lllite(
            pipeline.unet,
            cond_emb_dim=args.cond_emb_dim, mlp_dim=args.mlp_dim,
            cond_image_size=args.size, sample_size=latent_hw,
            inference_mode=True,
        )
        controller.load_state_dict(load_file(args.lllite_weights), strict=True)
        controller.to(device=args.device, dtype=torch.bfloat16)
        controller.eval()
        controller.set_multiplier(args.lllite_multiplier)
        print(f"[lllite] loaded  multiplier={args.lllite_multiplier}")

    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=args.steps, prompt=args.prompt,
    )
    inner = EditWorker(
        pipeline=pipeline, state=state, device=args.device,
        dtype=torch.bfloat16, init_mode="pure", seed=args.seed,
        compile=args.compile, compile_mode="reduce-overhead",
        lllite_controller=controller,
    )
    spec = SpeculativeEditWorker(
        inner=inner, flow_thresh=args.flow_thresh, max_consec=args.max_consec,
    )

    writer = None
    timings = []
    n_hits = 0
    n_total = 0
    print(f"[run]  flow_thresh={args.flow_thresh}  max_consec={args.max_consec}")
    t_start = time.perf_counter()
    for idx, frame, fps in iter_video_frames(args.input, args.size):
        if args.max_frames is not None and idx >= args.max_frames:
            break
        out_img, t = spec.step(frame)
        if writer is None:
            writer = VideoWriter(args.output, args.size, fps)
        writer.write_pil(out_img)
        timings.append(t)
        n_total += 1
        if t.accepted:
            n_hits += 1
        if idx % 10 == 0:
            tag = "HIT " if t.accepted else "MISS"
            print(
                f"[{idx:04d}] {tag} flow_max={t.flow_max:.1f}  total={t.total_ms:.0f}ms  "
                f"denoise={t.denoise_ms:.0f}  dec={t.vae_dec_ms:.0f}",
                flush=True,
            )
    if writer:
        writer.close()
    wall = (time.perf_counter() - t_start) * 1000

    if not timings:
        return
    n = len(timings)
    avg = lambda attr: sum(getattr(t, attr) for t in timings) / n
    hit_rate = n_hits / n_total if n_total else 0.0
    fps_step = n / (sum(t.total_ms for t in timings) / 1000)
    fps_wall = n / (wall / 1000)
    print(
        f"\nframes={n}  hit_rate={hit_rate*100:.1f}%  "
        f"fps_step={fps_step:.2f}  fps_wall={fps_wall:.2f}\n"
        f"  per-frame avg ms:  total={avg('total_ms'):.0f}  "
        f"denoise={avg('denoise_ms'):.0f}  dec={avg('vae_dec_ms'):.0f}  "
        f"te={avg('te_ms'):.0f}  enc={avg('vae_enc_ms'):.0f}"
    )


if __name__ == "__main__":
    main()
