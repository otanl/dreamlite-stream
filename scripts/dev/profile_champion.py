"""Profile champion path component-by-component on DAVIS blackswan.

Reports per-call wall-time for: prompt_encode (TE+image_to_latent),
UNet denoise step, VAE decode, and end-to-end fps. Use this to defend
the TE-bound claim with specific numbers in the paper.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402
from dreamlite_lllite.inject import apply_lllite  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from dreamlite_stream import pipeline_ops as ops  # noqa: E402
from dreamlite_stream.runtime import iter_video_frames  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights", default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--mp4", default=str(_ROOT / "assets" / "davis_mp4" / "blackswan.mp4"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_iters", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--no_lllite", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    # LLLite
    if not args.no_lllite:
        print(f"[lllite] {args.lllite_weights}")
        sd = load_file(args.lllite_weights)
        vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
        latent_hw = args.size // vae_downsample
        controller = apply_lllite(
            pipeline.unet, cond_emb_dim=32, mlp_dim=64,
            cond_image_size=args.size, sample_size=latent_hw,
            block_filter=["down_blocks"], inference_mode=True,
            max_batch_size=args.batch_size,
        )
        controller.load_state_dict(sd, strict=False)
        controller.to(device=args.device, dtype=torch.bfloat16)
        controller.eval()
        controller.set_multiplier(1.0)
        n_hooks = len(controller.modules_dict)
    else:
        controller = None
        n_hooks = 0

    # Compile UNet
    pipeline.unet = torch.compile(
        pipeline.unet, mode="reduce-overhead", fullgraph=False, dynamic=False,
    )

    # Load B=16 frames
    iterator = iter_video_frames(args.mp4, args.size)
    frames = []
    for idx, frame, _ in iterator:
        if len(frames) >= args.batch_size:
            break
        frames.append(frame)
    assert len(frames) == args.batch_size

    B = args.batch_size
    print(f"\n[setup] B={B} K={args.steps} size={args.size} hooks={n_hooks}")

    # Timesteps
    timesteps, _ = ops.set_timesteps(
        pipeline, height=args.size, width=args.size,
        num_inference_steps=args.steps, device=args.device,
    )
    time_ids = ops.make_time_ids(args.size, args.size, args.device, torch.bfloat16).expand(B, -1)

    # Helper: time a block
    def time_block(fn, n_iters: int, warmup: int):
        torch.cuda.synchronize()
        times = []
        for i in range(n_iters + warmup):
            t0 = time.perf_counter()
            r = fn()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000  # ms
            if i >= warmup:
                times.append(dt)
        return times, r

    # 1. TE encode_prompt_edit_batch (TE + image-to-latent + decorated prompt)
    print("\n[profile] timing TE batched encode...")
    def fn_te():
        return ops.encode_prompt_edit_batch(
            pipeline, prompts=[args.prompt] * B, images=frames,
            device=args.device, dtype=torch.bfloat16,
        )
    te_times, (prompt_embeds, prompt_mask) = time_block(fn_te, args.n_iters, args.warmup)
    print(f"  TE prompt encode (B={B}): {mean(te_times):.1f} ± {stdev(te_times):.1f} ms")

    # 1b. Image-to-latent encode (VAE encoder on input frame)
    print("\n[profile] timing image_to_latent_batch...")
    def fn_imglat():
        return ops.encode_image_to_latent_batch(
            pipeline, images=frames,
            height=args.size, width=args.size,
            device=args.device, dtype=torch.bfloat16,
        )
    imglat_times, ref_latents = time_block(fn_imglat, args.n_iters, args.warmup)
    print(f"  VAE encode (image->latent, B={B}): {mean(imglat_times):.1f} ± {stdev(imglat_times):.1f} ms")

    # Pre-create LLLite cond once (warped prev = current frame as fallback)
    if controller is not None:
        # for profiling: use input image as cond_image (worst case, same shape)
        cond_imgs = [np.asarray(f.convert("RGB")) if hasattr(f, "convert") else np.asarray(f) for f in frames]
        cond_imgs = np.stack(cond_imgs, axis=0)  # (B, H, W, 3)
        cond_t = (torch.from_numpy(cond_imgs).permute(0, 3, 1, 2).float() / 127.5 - 1.0).to(args.device, torch.bfloat16)
        controller.set_cond_image(cond_t)

    # 2. UNet denoise (K=1, B=16)
    print("\n[profile] timing UNet denoise (K=1, B=16)...")
    def fn_unet():
        # Reset scheduler each call (denoise advances sigmas internally)
        ts, _ = ops.set_timesteps(
            pipeline, height=args.size, width=args.size,
            num_inference_steps=args.steps, device=args.device,
        )
        init_latents = ops.make_init_noise(
            pipeline, height=args.size, width=args.size,
            device=args.device, dtype=torch.bfloat16, generator=None, batch_size=B,
        )
        return ops.denoise(
            pipeline,
            init_latents=init_latents,
            cond_image_latents=ref_latents,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            time_ids=time_ids,
            timesteps=ts,
        )
    unet_times, out_latents = time_block(fn_unet, args.n_iters, args.warmup)
    print(f"  UNet denoise (B={B}, K={args.steps}, hooks={n_hooks}): {mean(unet_times):.1f} ± {stdev(unet_times):.1f} ms")

    # 3. VAE decode
    print("\n[profile] timing VAE decode...")
    def fn_vae():
        return ops.decode_latent(pipeline, out_latents)
    vae_times, _ = time_block(fn_vae, args.n_iters, args.warmup)
    print(f"  VAE decode (B={B}): {mean(vae_times):.1f} ± {stdev(vae_times):.1f} ms")

    # Summary
    te_ms = mean(te_times)
    imglat_ms = mean(imglat_times)
    unet_ms = mean(unet_times)
    vae_ms = mean(vae_times)
    side_stream_ms = te_ms + imglat_ms  # both on side stream
    total_serial = te_ms + imglat_ms + unet_ms + vae_ms
    total_pipelined = max(side_stream_ms, unet_ms) + vae_ms
    fps_serial = (B * 1000) / total_serial
    fps_pipelined = (B * 1000) / total_pipelined

    print("\n========== summary ==========")
    print(f"{'component':<28} {'ms/batch':>12} {'ms/frame':>12} {'% of frame budget':>18}")
    print(f"{'TE prompt encode':<28} {te_ms:>12.1f} {te_ms/B:>12.2f} {te_ms/total_serial*100:>17.1f}%")
    print(f"{'VAE encode (img->latent)':<28} {imglat_ms:>12.1f} {imglat_ms/B:>12.2f} {imglat_ms/total_serial*100:>17.1f}%")
    print(f"{'  side-stream subtotal':<28} {side_stream_ms:>12.1f} {side_stream_ms/B:>12.2f} {side_stream_ms/total_serial*100:>17.1f}%")
    print(f"{'UNet denoise':<28} {unet_ms:>12.1f} {unet_ms/B:>12.2f} {unet_ms/total_serial*100:>17.1f}%")
    print(f"{'VAE decode':<28} {vae_ms:>12.1f} {vae_ms/B:>12.2f} {vae_ms/total_serial*100:>17.1f}%")
    print()
    print(f"{'serial total':<28} {total_serial:>12.1f} {total_serial/B:>12.2f} -> {fps_serial:.1f} FPS")
    print(f"{'pipelined (max(side,UN)+VAE)':<28} {total_pipelined:>12.1f} {total_pipelined/B:>12.2f} -> {fps_pipelined:.1f} FPS")
    print(f"{'side-stream floor':<28} {side_stream_ms:>12.1f} {side_stream_ms/B:>12.2f} -> {(B*1000)/side_stream_ms:.1f} FPS")
    print(f"{'UNet floor':<28} {unet_ms:>12.1f} {unet_ms/B:>12.2f} -> {(B*1000)/unet_ms:.1f} FPS")
    bottleneck = "TE/VAE-encode side-stream" if side_stream_ms > unet_ms else "UNet"
    print(f"\n[verdict] {bottleneck}-bound at this configuration.")


if __name__ == "__main__":
    main()
