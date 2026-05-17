"""Validate TRT engine: numerical match against PyTorch UNet, then time it."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402

from dreamlite_stream.trt_unet import TRTUNetWrapper  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--engine", default=str(_ROOT / "out" / "trt" / "unet_b8_512.engine"))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--prompt_seq_len", type=int, default=200)
    p.add_argument("--n_runs", type=int, default=20)
    p.add_argument("--n_warmup", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.float16

    print(f"[load] PyTorch  {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    unet = pipeline.unet
    unet.eval()

    print(f"[load] TRT engine  {args.engine}")
    trt_unet = TRTUNetWrapper(args.engine)
    print(f"  inputs:  {trt_unet.input_names}")
    print(f"  outputs: {trt_unet.output_names}")

    vae_scale = pipeline.vae_scale_factor
    lat_h = lat_w = args.size // vae_scale
    B = args.batch_size
    L = args.prompt_seq_len

    torch.manual_seed(42)
    model_input = torch.randn(B, 4, lat_h, lat_w * 2, device=device, dtype=dtype)
    timestep = torch.full((B,), 500.0, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(B, L, 2048, device=device, dtype=dtype)
    encoder_attention_mask = torch.ones(B, L, device=device, dtype=torch.long)
    time_ids = torch.tensor([[args.size, args.size]] * B, device=device, dtype=dtype)

    print("\n[numerical] PyTorch reference forward...")
    with torch.no_grad():
        ref_out = unet(
            model_input, timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            added_cond_kwargs={"time_ids": time_ids},
            return_dict=False,
        )[0]
    print(f"  ref_out.shape = {tuple(ref_out.shape)}, dtype={ref_out.dtype}")

    print("\n[numerical] TRT forward...")
    trt_out = trt_unet(
        model_input, timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        added_cond_kwargs={"time_ids": time_ids},
    )[0]
    print(f"  trt_out.shape = {tuple(trt_out.shape)}, dtype={trt_out.dtype}")

    diff = (ref_out.float() - trt_out.float()).abs()
    print(f"\n[diff]  max={diff.max().item():.4e}  mean={diff.mean().item():.4e}  "
          f"L1={diff.sum().item():.2f}")
    cos = torch.nn.functional.cosine_similarity(
        ref_out.flatten().float(), trt_out.flatten().float(), dim=0,
    ).item()
    print(f"  cosine_sim = {cos:.6f}")

    print("\n[speed] timing (n_warmup={}, n_runs={})".format(args.n_warmup, args.n_runs))
    # Warmup
    for _ in range(args.n_warmup):
        with torch.no_grad():
            _ = unet(
                model_input, timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                added_cond_kwargs={"time_ids": time_ids},
                return_dict=False,
            )
        _ = trt_unet(
            model_input, timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            added_cond_kwargs={"time_ids": time_ids},
        )
    torch.cuda.synchronize()

    # PyTorch
    t0 = time.perf_counter()
    for _ in range(args.n_runs):
        with torch.no_grad():
            _ = unet(
                model_input, timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                added_cond_kwargs={"time_ids": time_ids},
                return_dict=False,
            )[0]
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / args.n_runs * 1000

    # TRT
    t0 = time.perf_counter()
    for _ in range(args.n_runs):
        _ = trt_unet(
            model_input, timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            added_cond_kwargs={"time_ids": time_ids},
        )[0]
    torch.cuda.synchronize()
    trt_ms = (time.perf_counter() - t0) / args.n_runs * 1000

    print(f"  PyTorch eager fp16:   {pt_ms:.2f} ms/call  ({pt_ms/B:.2f} ms/frame)")
    print(f"  TensorRT fp16:        {trt_ms:.2f} ms/call  ({trt_ms/B:.2f} ms/frame)")
    print(f"  TRT speedup:          {pt_ms / trt_ms:.2f}x")


if __name__ == "__main__":
    main()
