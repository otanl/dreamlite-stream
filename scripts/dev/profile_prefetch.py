"""Profile encode_prompt_edit_batch sub-steps to find where the 320ms goes.

Runs the function directly (no NDI, no threading), 10 iterations, prints
ms for each phase: resize, processor, TE forward, hidden access, etc.

Compare to champion_eval.py's apparent per-iter wall (~270ms) to identify
which sub-step is the regression.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

from dreamlite import DreamLiteMobilePipeline


def main():
    device = "cuda"
    dtype = torch.bfloat16
    B = 8
    SIZE = 512

    print("[load] pipeline ...")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        str(_DREAMLITE / "models" / "DreamLite-mobile"),
        torch_dtype=dtype,
    ).to(device)

    # Synthetic batch of B 512x512 PIL images + 1 prompt
    print("[setup] B=8 input batch")
    prompt = "transfer this to oil painting style, vibrant colors"
    images = [
        Image.fromarray((np.random.rand(SIZE, SIZE, 3) * 255).astype(np.uint8))
        for _ in range(B)
    ]
    prompts = [prompt] * B

    template = (
        "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
        "texture, objects, background), then explain how the user's text instruction should alter "
        "or modify the image. Generate a new image that meets the user's requirements while maintaining "
        "consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
    )
    decorated = [
        f"[Edit]: A diptych with two side-by-side images of the same scene. "
        f"Compared to the right side, the left one has {p}"
        for p in prompts
    ]
    txts = [template.format(p) for p in decorated]
    drop_idx = 64

    def time_block():
        t = time.perf_counter()
        return lambda: (time.perf_counter() - t) * 1000

    print("\n=== 10 iterations (after 2 warmup) ===\n")
    print(f"{'iter':>4s} {'resize':>8s} {'processor':>10s} {'.to(dev)':>9s} "
          f"{'TE.fwd':>8s} {'hidden':>8s} {'mask_loop':>10s} {'total':>8s}")

    for i in range(12):
        torch.cuda.synchronize()
        T = time_block()
        # 1. resize images to 256x256 for Q3-VL
        t = time_block()
        pil_imgs = [img.resize((256, 256), Image.Resampling.LANCZOS) for img in images]
        t_resize = t()

        # 2. HF processor (tokenize + image preprocessing)
        t = time_block()
        tk_out_cpu = pipeline.processor(
            text=txts, images=pil_imgs, padding=True, return_tensors="pt",
        )
        t_proc = t()

        # 3. .to(device)
        t = time_block()
        tk_out = tk_out_cpu.to(device)
        torch.cuda.synchronize()  # measure transfer
        t_to = t()

        # 4. TE forward
        t = time_block()
        outputs = pipeline.text_encoder(
            input_ids=tk_out.input_ids,
            attention_mask=tk_out.attention_mask,
            pixel_values=tk_out.pixel_values,
            image_grid_thw=tk_out.image_grid_thw,
            output_hidden_states=True,
        )
        torch.cuda.synchronize()  # measure GPU
        t_te = t()

        # 5. hidden states access / split
        t = time_block()
        hidden_states = outputs.hidden_states[-1]
        split = pipeline._extract_masked_hidden(hidden_states, tk_out.attention_mask)
        split = [e[drop_idx:] for e in split]
        from torch.nn.utils.rnn import pad_sequence
        prompt_embeds = pad_sequence(split, batch_first=True, padding_value=0).to(
            dtype=dtype, device=device,
        )
        torch.cuda.synchronize()
        t_hidden = t()

        # 6. mask loop
        t = time_block()
        Bp, Lp, _ = prompt_embeds.shape
        prompt_embeds_mask = torch.zeros((Bp, Lp), dtype=torch.long, device=device)
        for j, seq in enumerate(split):
            prompt_embeds_mask[j, : seq.shape[0]] = 1
        torch.cuda.synchronize()
        t_mask = t()

        t_total = T()

        if i >= 2:  # skip warmup
            print(f"{i:>4d} {t_resize:>7.1f}ms {t_proc:>9.1f}ms {t_to:>8.1f}ms "
                  f"{t_te:>7.1f}ms {t_hidden:>7.1f}ms {t_mask:>9.1f}ms {t_total:>7.1f}ms")

    print("\nDONE")


if __name__ == "__main__":
    main()
