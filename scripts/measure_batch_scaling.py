"""Measure UNet forward time at batch=1, 2, 4, 8.

If batch=N takes less than N x batch=1 time, the GPU SMs aren't saturated
at batch=1 and batched inference will give "free" throughput. This is the
key empirical question for whether multi-frame parallelism on a single GPU
is worth the engineering investment.

Reports:
    sub_linearity = T_at_N / (N * T_at_1)
        1.0  = perfectly linear (no benefit from batching, SMs saturated)
        0.7  = ~1.4x throughput per GPU-second from batching
        0.5  = ~2x throughput (effectively perfect parallelism)
        <0.5 = batch overhead pays for itself many times over
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import torch
import torch._dynamo

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n_runs", type=int, default=20)
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--compile", action="store_true",
                   help="compile UNet (per-batch-size; pays compile cost N times)")
    return p.parse_args()


@torch.no_grad()
def measure_one(unet, B: int, lat_h: int, lat_w: int, dtype, device,
                n_warmup: int, n_runs: int) -> float:
    # DreamLite spatial-concat: model_input = cat([noisy, cond], dim=W) -> width 2x
    model_input = torch.randn(B, 4, lat_h, lat_w * 2, device=device, dtype=dtype)
    # Plausible prompt embeds (length 200 is what DreamLite uses for edit mode)
    L = 200
    prompt_embeds = torch.randn(B, L, 2048, device=device, dtype=dtype)
    prompt_mask = torch.ones(B, L, device=device, dtype=torch.long)
    time_ids = torch.tensor([[lat_w * 8, lat_h * 8]] * B, device=device, dtype=dtype)
    t = torch.tensor([500.0] * B, device=device, dtype=dtype)

    for _ in range(n_warmup):
        _ = unet(
            model_input, timestep=t,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask,
            added_cond_kwargs={"time_ids": time_ids},
            return_dict=False,
        )[0]
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = unet(
            model_input, timestep=t,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask,
            added_cond_kwargs={"time_ids": time_ids},
            return_dict=False,
        )[0]
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000  # ms per call


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=dtype,
    ).to(device)

    if args.compile:
        torch._dynamo.config.cache_size_limit = 64
        pipeline.unet = torch.compile(
            pipeline.unet, mode="reduce-overhead",
            fullgraph=False, dynamic=False,
        )
        print("[compile] unet wrapped (each batch size pays its own compile cost)")

    vae_scale = pipeline.vae_scale_factor
    lat_h = lat_w = args.size // vae_scale

    print(
        f"\nMeasuring UNet forward at {args.size}x{args.size} "
        f"(latent {lat_h}x{lat_w*2} after spatial concat)\n"
    )
    print(f"{'batch':>6s}  {'ms/call':>8s}  {'ms/frame':>9s}  {'sub_linear':>10s}  "
          f"{'throughput':>11s}")
    print("-" * 60)

    t1 = None
    for B in args.batches:
        try:
            ms = measure_one(
                pipeline.unet, B, lat_h, lat_w, dtype, device,
                args.n_warmup, args.n_runs,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>6d}  OOM")
            continue
        if t1 is None:
            t1 = ms
            sub = 1.0
            tput = 1.0
        else:
            sub = ms / (B * t1)  # 1.0 = linear scaling, <1.0 = sub-linear
            tput = B / (ms / t1)  # frames per second per t1-batched second
        print(
            f"{B:>6d}  {ms:>8.1f}  {ms/B:>9.1f}  {sub:>10.3f}  {tput:>10.2f}x"
        )

    print(
        "\nInterpretation:\n"
        "  sub_linear < 0.7 -> batched inference gives >=1.4x throughput\n"
        "  sub_linear ~ 1.0 -> SMs are saturated; batching doesn't help\n"
    )


if __name__ == "__main__":
    main()
