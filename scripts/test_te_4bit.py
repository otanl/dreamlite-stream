"""Test 4-bit quantization of Qwen3-VL TE.

DreamLite paper uses 4-bit Qwen-VL for iPhone deployment to fit memory and
speed constraints. We test if it brings a 1.5-3x speedup on a 3090Ti while
preserving inference quality.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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
    p.add_argument("--quant", choices=["bf16", "fp16", "nf4", "fp4", "int8"], default="nf4")
    return p.parse_args()


def load_pipeline_with_quant_te(model_path: str, quant: str, device: str):
    from transformers import BitsAndBytesConfig

    if quant in ("bf16", "fp16"):
        dtype = torch.bfloat16 if quant == "bf16" else torch.float16
        return DreamLiteMobilePipeline.from_pretrained(model_path, torch_dtype=dtype).to(device), dtype

    if quant == "int8":
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
    else:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=quant,
            bnb_4bit_use_double_quant=True,
        )

    # Load pipeline normally (fp16) for everything except TE
    pipe = DreamLiteMobilePipeline.from_pretrained(
        model_path, torch_dtype=torch.float16,
    )
    # Re-load TE with quantization
    from transformers import Qwen3VLForConditionalGeneration
    te_path = Path(model_path) / "text_encoder"
    print(f"  loading TE from {te_path} with quantization={quant}...")
    te_quant = Qwen3VLForConditionalGeneration.from_pretrained(
        te_path,
        quantization_config=qconfig,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    # Swap into pipeline
    del pipe.text_encoder
    pipe.text_encoder = te_quant
    # Move other components to device (TE already there)
    pipe.vae = pipe.vae.to(device)
    pipe.unet = pipe.unet.to(device)
    return pipe, torch.float16


def make_inputs(B: int):
    prompts = [f"transfer this to oil painting style, vibrant colors (variant {i})" for i in range(B)]
    images = [
        Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
        for _ in range(B)
    ]
    return prompts, images


def main():
    args = parse_args()
    device = "cuda"

    print(f"[load] {args.model} with TE quant={args.quant}")
    pipe, dtype = load_pipeline_with_quant_te(args.model, args.quant, device)
    print(f"  TE class: {type(pipe.text_encoder).__name__}")
    n_params = sum(p.numel() for p in pipe.text_encoder.parameters())
    print(f"  TE params: {n_params/1e9:.2f}B")
    # Estimate VRAM for TE
    vram = 0
    for p in pipe.text_encoder.parameters():
        vram += p.element_size() * p.numel()
    print(f"  TE VRAM (rough): {vram/1024/1024/1024:.2f} GB")

    prompts, images = make_inputs(args.batch_size)

    print(f"\n[timing] B={args.batch_size}")
    for _ in range(args.n_warmup):
        ops.encode_prompt_edit_batch(pipe, prompts, images, device, dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.n_runs):
        ops.encode_prompt_edit_batch(pipe, prompts, images, device, dtype)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / args.n_runs * 1000
    print(f"  {args.quant}: {ms:.1f} ms/call ({ms/args.batch_size:.1f} ms/frame)")


if __name__ == "__main__":
    main()
