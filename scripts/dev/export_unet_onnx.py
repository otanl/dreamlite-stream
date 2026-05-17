"""Export DreamLite-mobile UNet to ONNX.

Fixed shapes for the inference path used by mvp3_batched:
  batch_size = 8 (configurable)
  size       = 512 (configurable)
  steps      = 1   (we run 1-step at inference; affects timestep tensor shape)
  prompt seq = 200 (max_sequence_length used by DreamLite TE)
  cross_dim  = 2304 (UNet's encoder_hid_proj output dim)

bf16 weights are temporarily cast to fp16 for ONNX export (TRT doesn't always
have great bf16 kernels on consumer GPUs; we'll convert back if needed).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--out", default=str(_ROOT / "out" / "trt" / "unet_b8_512.onnx"))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--prompt_seq_len", type=int, default=200)
    p.add_argument("--encoder_hid_dim", type=int, default=2048,
                   help="Qwen3-VL hidden dim BEFORE encoder_hid_proj projection")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")

    print(f"[load] {args.model} (dtype={args.dtype})")
    pipeline = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    unet = pipeline.unet
    unet.eval()

    vae_scale = pipeline.vae_scale_factor
    lat_h = lat_w = args.size // vae_scale
    B = args.batch_size

    # DreamLite spatial-concat: model_input width is doubled at the outer layer
    model_input = torch.randn(B, 4, lat_h, lat_w * 2, device=device, dtype=dtype)
    timestep = torch.full((B,), 500.0, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(B, args.prompt_seq_len, args.encoder_hid_dim, device=device, dtype=dtype)
    encoder_attention_mask = torch.ones(B, args.prompt_seq_len, device=device, dtype=torch.long)
    time_ids = torch.tensor([[args.size, args.size]] * B, device=device, dtype=dtype)

    print(f"[shapes]")
    print(f"  model_input         : {tuple(model_input.shape)} {model_input.dtype}")
    print(f"  timestep            : {tuple(timestep.shape)} {timestep.dtype}")
    print(f"  encoder_hidden      : {tuple(encoder_hidden_states.shape)} {encoder_hidden_states.dtype}")
    print(f"  encoder_attn_mask   : {tuple(encoder_attention_mask.shape)}")
    print(f"  time_ids            : {tuple(time_ids.shape)} {time_ids.dtype}")

    # Wrap UNet so it has a positional-arg signature compatible with onnx.export.
    class _UNetWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, model_input, timestep, encoder_hidden_states, encoder_attention_mask, time_ids):
            return self.m(
                model_input,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                added_cond_kwargs={"time_ids": time_ids},
                return_dict=False,
            )[0]

    wrapper = _UNetWrapper(unet).to(device).eval()

    # Sanity-check forward
    print("[sanity] running PyTorch forward...")
    with torch.no_grad():
        ref_out = wrapper(model_input, timestep, encoder_hidden_states, encoder_attention_mask, time_ids)
    print(f"  ref_out.shape = {tuple(ref_out.shape)}, dtype={ref_out.dtype}")

    print(f"\n[export] writing {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (model_input, timestep, encoder_hidden_states, encoder_attention_mask, time_ids),
            str(out_path),
            input_names=["model_input", "timestep", "encoder_hidden_states",
                         "encoder_attention_mask", "time_ids"],
            output_names=["noise_pred"],
            opset_version=17,
            do_constant_folding=True,
            # Static shapes (no dynamic_axes) — TRT prefers fixed shapes for
            # best optimization. We can re-export with different B/size if
            # we need flexibility later.
        )

    print(f"[done] saved ONNX: {out_path}")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  file size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
