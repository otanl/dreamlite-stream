"""Measure Qwen3-VL TE forward time at batch=1, 2, 4, 8 in [Edit] mode.

Per-frame [Edit] TE includes vision-token computation over a 256x256 image —
this is the dominant non-UNet cost in our streaming pipeline (~150 ms/frame
in eager mode at batch=1).

If TE scales sub-linearly with batch, we can hide it inside the UNet pipeline
even when the UNet itself is well-saturated by compile.
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--mode", choices=["edit", "generate"], default="edit",
                   help="edit = vision+text (heavy); generate = text-only (light)")
    return p.parse_args()


@torch.no_grad()
def measure_one(pipe, B: int, mode: str, device, dtype,
                n_warmup: int, n_runs: int) -> float:
    """Build B DIFFERENT image+prompt pairs (representative of streaming) and
    drive the processor + TE directly with the batch."""
    from torch.nn.utils.rnn import pad_sequence
    base_prompt = (
        "[Edit]: A diptych with two side-by-side images of the same scene. "
        "Compared to the right side, the left one has transfer this to oil painting style"
    )

    if mode == "generate":
        prompts = [f"[Generate]: oil painting variant {i}" for i in range(B)]

        def fwd():
            _, _ = pipe.encode_prompt(
                mode="generate", prompts=prompts,
                device=device, dtype=dtype,
            )
    else:
        prompts = [f"{base_prompt} (variant {i})" for i in range(B)]
        # B independently sampled noise images, resized to 256x256 (TE input size)
        images = [
            Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
            for _ in range(B)
        ]
        # Recreate the inner steps of pipe.encode_prompt(mode="edit") but with
        # one independent image per prompt rather than the one-image-broadcast
        # the public API uses.
        template = (
            "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
            "texture, objects, background), then explain how the user's text instruction should alter "
            "or modify the image. Generate a new image that meets the user's requirements while maintaining "
            "consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        )
        txts = [template.format(p) for p in prompts]

        def fwd():
            tk_out = pipe.processor(
                text=txts, images=images, padding=True, return_tensors="pt",
            ).to(device)
            outputs = pipe.text_encoder(
                input_ids=tk_out.input_ids,
                attention_mask=tk_out.attention_mask,
                pixel_values=tk_out.pixel_values,
                image_grid_thw=tk_out.image_grid_thw,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            split = pipe._extract_masked_hidden(hidden_states, tk_out.attention_mask)
            split = [e[64:] for e in split]
            return pad_sequence(split, batch_first=True, padding_value=0).to(dtype=dtype, device=device)

    for _ in range(n_warmup):
        fwd()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        fwd()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"[load] {args.model}")
    pipe = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=dtype,
    ).to(device)

    print(f"\nMeasuring TE forward in {args.mode} mode\n")
    print(f"{'batch':>6s}  {'ms/call':>8s}  {'ms/frame':>9s}  "
          f"{'sub_linear':>10s}  {'throughput':>11s}")
    print("-" * 60)

    t1 = None
    for B in args.batches:
        try:
            ms = measure_one(
                pipe, B, args.mode, device, dtype,
                args.n_warmup, args.n_runs,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>6d}  OOM")
            continue
        if t1 is None:
            t1 = ms
            sub, tput = 1.0, 1.0
        else:
            sub = ms / (B * t1)
            tput = B / (ms / t1)
        print(
            f"{B:>6d}  {ms:>8.1f}  {ms/B:>9.1f}  {sub:>10.3f}  {tput:>10.2f}x"
        )


if __name__ == "__main__":
    main()
