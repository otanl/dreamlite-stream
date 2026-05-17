"""Pinpoint where TRT output drifts from PyTorch in actual pipeline conditions.

Random-input UNet test showed cos_sim 0.9996 (near-perfect). But in-pipeline
output is visibly different. This script:
  1. Runs the real encode_prompt / encode_image_to_latent path
  2. Builds a real model_input
  3. Compares PyTorch UNet vs TRT UNet noise_pred element-wise
  4. Decodes both noise_preds (= -latent for 1-step) through VAE
  5. Reports diff statistics at each stage
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import cv2
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
from dreamlite_stream.trt_unet import TRTUNetWrapper  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--engine", default=str(_ROOT / "out" / "trt" / "unet_b8_512.engine"))
    p.add_argument("--video", default=str(_ROOT / "assets" / "davis_mp4" / "dance-twirl.mp4"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    return p.parse_args()


def stat(name: str, t: torch.Tensor):
    f = t.float()
    print(f"  {name:30s}  shape={tuple(t.shape)}  "
          f"min={f.min().item():.4f}  max={f.max().item():.4f}  "
          f"mean={f.mean().item():.4f}  std={f.std().item():.4f}")


def diff_stats(name: str, a: torch.Tensor, b: torch.Tensor):
    d = (a.float() - b.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0,
    ).item()
    print(f"  {name:30s}  diff: max={d.max().item():.4f}  "
          f"mean={d.mean().item():.6f}  median={d.median().item():.6f}  "
          f"cos_sim={cos:.6f}")


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.float16
    B = args.batch_size

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    pyt_unet = pipeline.unet
    pyt_unet.eval()
    print(f"[load] TRT engine {args.engine}")
    trt_unet = TRTUNetWrapper(args.engine)
    print(f"  engine seq_len: {trt_unet.engine_seq_len}")

    # --- Real pipeline inputs ---
    cap = cv2.VideoCapture(args.video)
    pil_frames = []
    for _ in range(B):
        ok, bgr = cap.read()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        s = min(h, w); y, x = (h - s) // 2, (w - s) // 2
        rgb = cv2.resize(rgb[y:y+s, x:x+s], (args.size, args.size), interpolation=cv2.INTER_AREA)
        pil_frames.append(Image.fromarray(rgb))
    cap.release()

    prompts = [args.prompt] * B
    print(f"\n[encode_prompt_edit]  L tokens varies per batch")
    with torch.no_grad():
        prompt_embeds, prompt_mask = ops.encode_prompt_edit_batch(
            pipeline, prompts, pil_frames, device, dtype,
        )
    print(f"  prompt_embeds shape: {tuple(prompt_embeds.shape)}")
    print(f"  prompt_mask  shape: {tuple(prompt_mask.shape)}")
    stat("prompt_embeds", prompt_embeds)

    print(f"\n[encode_image]")
    with torch.no_grad():
        ref_latents = ops.encode_image_to_latent_batch(
            pipeline, pil_frames, args.size, args.size, device, dtype,
        )
    stat("ref_latents", ref_latents)

    # 1-step setup
    timesteps, _ = ops.set_timesteps(pipeline, args.size, args.size, 1, device)
    print(f"\n[scheduler] timesteps: {timesteps.cpu().tolist()}")

    g = torch.Generator(device="cpu").manual_seed(42)
    init_noise = torch.randn(B, 4, args.size // 8, args.size // 8, dtype=dtype, generator=g).to(device)
    stat("init_noise", init_noise)

    model_input = torch.cat([init_noise, ref_latents], dim=3)
    stat("model_input (cat)", model_input)
    time_ids = ops.make_time_ids(args.size, args.size, device, dtype).expand(B, -1)
    t = timesteps[0]
    timestep = t.expand(B).to(dtype)

    # --- PyTorch UNet ---
    print(f"\n[PyTorch UNet]")
    with torch.no_grad():
        noise_pred_pyt = pyt_unet(
            model_input, timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask,
            added_cond_kwargs={"time_ids": time_ids},
            return_dict=False,
        )[0]
    stat("noise_pred_pyt", noise_pred_pyt)

    # --- TRT UNet ---
    print(f"\n[TRT UNet]")
    noise_pred_trt = trt_unet(
        model_input, timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_mask,
        added_cond_kwargs={"time_ids": time_ids},
    )[0]
    torch.cuda.synchronize()
    stat("noise_pred_trt", noise_pred_trt)

    # --- Diff at noise_pred level ---
    print(f"\n[noise_pred diff stats]")
    diff_stats("FULL output", noise_pred_pyt, noise_pred_trt)
    # Slice as denoise() does
    nps_pyt = noise_pred_pyt[..., : init_noise.shape[-1]]
    nps_trt = noise_pred_trt[..., : init_noise.shape[-1]]
    print(f"\n[noise_pred diff stats (sliced as in pipeline)]")
    diff_stats("SLICED output", nps_pyt, nps_trt)
    stat("nps_pyt", nps_pyt)
    stat("nps_trt", nps_trt)

    # --- Apply scheduler step ---
    print(f"\n[scheduler.step]")
    pipeline.scheduler.set_timesteps(timesteps=timesteps.cpu().numpy().tolist(), device=device)
    latents_pyt = pipeline.scheduler.step(nps_pyt, t, init_noise.clone(), return_dict=False)[0]
    pipeline.scheduler.set_timesteps(timesteps=timesteps.cpu().numpy().tolist(), device=device)
    latents_trt = pipeline.scheduler.step(nps_trt, t, init_noise.clone(), return_dict=False)[0]
    stat("latents_pyt", latents_pyt)
    stat("latents_trt", latents_trt)
    diff_stats("latent diff", latents_pyt, latents_trt)

    # --- VAE decode both ---
    print(f"\n[VAE decode]")
    with torch.no_grad():
        img_pyt = ops.decode_latent(pipeline, latents_pyt, output_type="np")[0]
        img_trt = ops.decode_latent(pipeline, latents_trt, output_type="np")[0]
    print(f"  img_pyt shape: {img_pyt.shape}, dtype: {img_pyt.dtype}")
    print(f"  img_trt shape: {img_trt.shape}, dtype: {img_trt.dtype}")
    img_pyt_u8 = (img_pyt * 255).astype(np.uint8)
    img_trt_u8 = (img_trt * 255).astype(np.uint8)
    img_diff = np.abs(img_pyt_u8.astype(np.float32) - img_trt_u8.astype(np.float32)).mean()
    print(f"  decoded image L1 diff: {img_diff:.2f}  (range 0-255)")

    # Save side-by-side
    side = np.concatenate([
        cv2.cvtColor(img_pyt_u8, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(img_trt_u8, cv2.COLOR_RGB2BGR),
    ], axis=1)
    out_path = _ROOT / "out" / "sweep" / "trt_drift_b0.png"
    cv2.imwrite(str(out_path), side)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
