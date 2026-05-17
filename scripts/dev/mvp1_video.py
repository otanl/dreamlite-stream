"""MVP-1: streaming video edit via a single DreamLite-mobile worker.

Usage:
    python scripts/mvp1_video.py \
        --input path/to/clip.mp4 \
        --output out/mvp1.mp4 \
        --prompt "transfer this to oil-painting style" \
        --size 512 --steps 4 --init pure
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch

# Make sibling repos importable without pip-installing.
_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402

from dreamlite_stream import EditWorker, SharedState  # noqa: E402
from dreamlite_stream.runtime import run_video, run_video_pipelined  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="out/mvp1.mp4")
    p.add_argument("--prompt", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--init", default="pure", choices=["pure", "prev"],
        help="init noise mode (pure = randn each frame; prev = blend with previous latent)",
    )
    p.add_argument("--noise_strength", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument(
        "--compile", action="store_true",
        help="torch.compile(unet) for ~2x denoise speedup (first frame is slow)",
    )
    p.add_argument(
        "--compile_mode", default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    p.add_argument(
        "--compile_backend", default=None,
        choices=[None, "cudagraphs", "aot_eager", "inductor"],
        help="override compile backend (None = use --compile_mode w/ inductor; cudagraphs = no Triton needed)",
    )
    p.add_argument(
        "--pipelined", action="store_true",
        help="overlap TE+VAE_enc of next frame with denoise of current frame",
    )
    return p.parse_args()


def main():
    args = parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"[load] {args.model} (dtype={args.dtype})")
    pipeline = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=dtype).to(args.device)

    state = SharedState(
        height=args.size, width=args.size, num_inference_steps=args.steps, prompt=args.prompt,
    )
    worker = EditWorker(
        pipeline=pipeline, state=state, device=args.device, dtype=dtype,
        init_mode=args.init, noise_strength=args.noise_strength, seed=args.seed,
        compile=args.compile, compile_mode=args.compile_mode,
        compile_backend=args.compile_backend,
    )

    runner = run_video_pipelined if args.pipelined else run_video
    print(
        f"[run]  input={args.input} output={args.output} "
        f"size={args.size}x{args.size} steps={args.steps} init={args.init}"
        f" compile={args.compile} pipelined={args.pipelined}"
    )
    stats = runner(
        worker, in_path=args.input, out_path=args.output, size=args.size,
        max_frames=args.max_frames,
    )
    print(stats.report())


if __name__ == "__main__":
    main()
