"""Try torch.compile on the Qwen3-VL TE to see if it gives a quick win
before committing to TRT export work.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
from PIL import Image

torch._dynamo.config.cache_size_limit = 64

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402

from dreamlite_stream import pipeline_ops as ops  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--compile_mode", default="reduce-overhead")
    return p.parse_args()


def make_inputs(B: int, device, dtype):
    prompts = [f"transfer this to oil painting style, vibrant colors (variant {i})" for i in range(B)]
    images = [
        Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
        for _ in range(B)
    ]
    return prompts, images


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"[load] {args.model}")
    pipe = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)

    prompts, images = make_inputs(args.batch_size, device, dtype)

    # ---- Eager baseline ----
    print(f"\n[eager] B={args.batch_size}")
    for _ in range(args.n_warmup):
        ops.encode_prompt_edit_batch(pipe, prompts, images, device, dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.n_runs):
        ops.encode_prompt_edit_batch(pipe, prompts, images, device, dtype)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / args.n_runs * 1000
    print(f"  {eager_ms:.1f} ms/call ({eager_ms/args.batch_size:.1f} ms/frame)")

    # ---- compile attempt ----
    print(f"\n[compile] wrapping TE with torch.compile (mode={args.compile_mode})...")
    try:
        # Compile only the language model part, not the whole class which has
        # generate() etc. Most time is in the language model forward.
        pipe.text_encoder.language_model = torch.compile(
            pipe.text_encoder.language_model,
            mode=args.compile_mode, fullgraph=False, dynamic=False,
        )
        print("  compiled language_model")
    except AttributeError:
        # Different attribute names possible
        print("  AttributeError for language_model; trying full text_encoder...")
        try:
            pipe.text_encoder = torch.compile(
                pipe.text_encoder,
                mode=args.compile_mode, fullgraph=False, dynamic=False,
            )
            print("  compiled text_encoder")
        except Exception as e:
            print(f"  WARN: compile failed: {e}")
            return

    print(f"\n[compile timing] (warmup includes JIT cost, can be slow)")
    for i in range(args.n_warmup):
        t0 = time.perf_counter()
        ops.encode_prompt_edit_batch(pipe, prompts, images, device, dtype)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        print(f"  warmup {i}: {ms:.0f} ms")
    t0 = time.perf_counter()
    for _ in range(args.n_runs):
        ops.encode_prompt_edit_batch(pipe, prompts, images, device, dtype)
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) / args.n_runs * 1000
    print(f"\n[compile] {compile_ms:.1f} ms/call ({compile_ms/args.batch_size:.1f} ms/frame)")
    print(f"[speedup] {eager_ms / compile_ms:.2f}x")


if __name__ == "__main__":
    main()
