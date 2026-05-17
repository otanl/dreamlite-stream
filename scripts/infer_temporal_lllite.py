"""Inference with the temporal LLLite adapter.

Loads a trained adapter via dreamlite_lllite.apply_lllite() before any
torch.compile so the compiled graph includes the LLLite Linear hooks. Per
frame, the runtime computes optical flow from the previous input to the
current input, warps the previous DECODED output by it, and feeds the
warped image to controller.set_cond_image() — the trained adapter then
injects a temporal-consistency δ into the UNet attention.

Usage:
    python scripts/infer_temporal_lllite.py \
        --weights runs/temporal_lllite_v1/temporal_lllite_step000420.safetensors \
        --input  assets/davis_mp4/dance-twirl.mp4 \
        --output out/temporal_lllite.mp4 \
        --prompt "transfer this to oil painting style, vibrant colors" \
        --size 512 --steps 4 --multiplier 1.0 --pipelined
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch._dynamo
from safetensors.torch import load_file

# Some DreamLite resnet variations push the dynamo cache past its default
# 8-entry limit; raise it so all variants stay compiled.
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

from dreamlite_stream import EditWorker, SharedState  # noqa: E402
from dreamlite_stream.runtime import run_video, run_video_pipelined  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--weights", required=True, help="trained .safetensors checkpoint")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="out/temporal_lllite.mp4")
    p.add_argument("--prompt", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--multiplier", type=float, default=1.0,
                   help="LLLite multiplier; 0 = adapter off (= MVP-1 baseline), 1 = full")
    p.add_argument("--cond_emb_dim", type=int, default=32)
    p.add_argument("--mlp_dim", type=int, default=64)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--pipelined", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    # Attach LLLite BEFORE torch.compile so the compiled graph sees the hooks.
    vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
    latent_hw = args.size // vae_downsample
    print(f"[lllite] attaching adapter (cond_emb_dim={args.cond_emb_dim} mlp_dim={args.mlp_dim})")
    controller = apply_lllite(
        pipeline.unet,
        cond_emb_dim=args.cond_emb_dim,
        mlp_dim=args.mlp_dim,
        cond_image_size=args.size,
        sample_size=latent_hw,
        inference_mode=True,  # branchless forward → torch.compile compatible
    )
    sd = load_file(args.weights)
    controller.load_state_dict(sd, strict=True)
    controller.to(device=args.device, dtype=torch.bfloat16)
    controller.eval()
    controller.set_multiplier(args.multiplier)
    n = controller.num_parameters()
    print(f"  loaded {args.weights}  params={n/1e6:.2f}M  multiplier={args.multiplier}")

    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=args.steps, prompt=args.prompt,
    )
    worker = EditWorker(
        pipeline=pipeline, state=state, device=args.device,
        dtype=torch.bfloat16, init_mode="pure", seed=args.seed,
        compile=args.compile, compile_mode="reduce-overhead",
        lllite_controller=controller,
    )

    runner = run_video_pipelined if args.pipelined else run_video
    print(
        f"[run]  steps={args.steps} compile={args.compile} pipelined={args.pipelined} "
        f"mult={args.multiplier}"
    )
    stats = runner(
        worker, in_path=args.input, out_path=args.output, size=args.size,
        max_frames=args.max_frames,
    )
    print(stats.report())


if __name__ == "__main__":
    main()
