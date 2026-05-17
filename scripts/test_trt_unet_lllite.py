"""Numerical + speed test for LLLite-baked TRT engine.

Compares PyTorch (UNet + LLLite hooks) to TRT engine output on a real frame
batch. Reports cos_sim and timing.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
_LLLITE = _ROOT.parent / "dreamlite-lllite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))
sys.path.insert(0, str(_LLLITE / "src"))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402
from dreamlite_lllite import apply_lllite  # noqa: E402

from dreamlite_stream import pipeline_ops as ops  # noqa: E402
from dreamlite_stream.trt_unet_lllite import TRTUNetLLLiteWrapper  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--engine", default=str(_ROOT / "out" / "trt" / "unet_lllite_b8_512_dryrun.engine"))
    p.add_argument("--hooks_json", default=str(_ROOT / "out" / "trt" / "unet_lllite_b8_512_dryrun.hooks.json"))
    p.add_argument("--lllite_weights",
                   default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--video", default=str(_ROOT / "assets" / "davis_mp4" / "dance-twirl.mp4"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--n_warmup", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.float16
    B = args.batch_size

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    unet = pipeline.unet
    unet.eval()

    # Apply LLLite (matches export-time settings)
    vae_scale = pipeline.vae_scale_factor
    latent_hw = args.size // vae_scale
    controller = apply_lllite(
        unet, cond_emb_dim=32, mlp_dim=64,
        cond_image_size=args.size, sample_size=latent_hw,
        inference_mode=True, max_batch_size=B,
        block_filter=["down_blocks"],
    )
    sd = load_file(args.lllite_weights)
    controller.load_state_dict(sd, strict=False)
    controller.to(device=device, dtype=dtype)
    controller.eval()
    controller.set_multiplier(1.0)
    print(f"[lllite] {len(controller.modules_dict)} hooks")

    # Load TRT engine
    print(f"[load TRT] {args.engine}")
    trt_unet = TRTUNetLLLiteWrapper(
        engine_path=args.engine, hooks_json=args.hooks_json,
        controller=controller, device="cuda",
    )

    # Prepare a real batch of frames
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

    # Encode prompt + image
    prompts = [args.prompt] * B
    with torch.no_grad():
        prompt_embeds, prompt_mask = ops.encode_prompt_edit_batch(
            pipeline, prompts, pil_frames, device, dtype,
        )
        ref_latents = ops.encode_image_to_latent_batch(
            pipeline, pil_frames, args.size, args.size, device, dtype,
        )

    timesteps, _ = ops.set_timesteps(pipeline, args.size, args.size, 1, device)
    g = torch.Generator(device="cpu").manual_seed(42)
    init_noise = torch.randn(B, 4, latent_hw, latent_hw, dtype=dtype, generator=g).to(device)
    model_input = torch.cat([init_noise, ref_latents], dim=3)
    time_ids = ops.make_time_ids(args.size, args.size, device, dtype).expand(B, -1)
    t = timesteps[0]
    timestep = t.expand(B).to(dtype)

    # Cond image: use input frames as cond_image (matches train-time usage when prev is unknown)
    cond_image_np = np.stack([np.asarray(f.convert("RGB")) for f in pil_frames], axis=0).astype(np.float32)
    cond_image_np = cond_image_np / 255.0 * 2.0 - 1.0
    cond_image = torch.from_numpy(cond_image_np).permute(0, 3, 1, 2).contiguous().to(device, dtype)

    # PyTorch reference (LLLite-equipped via monkey-patch on host.forward)
    controller.set_cond_image(cond_image)
    with torch.no_grad():
        noise_pred_pyt = unet(
            model_input, timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask,
            added_cond_kwargs={"time_ids": time_ids},
            return_dict=False,
        )[0]
    print(f"[pyt] noise_pred shape: {tuple(noise_pred_pyt.shape)}")

    # TRT (cond_image set on the wrapper)
    trt_unet.set_cond_image(cond_image)
    noise_pred_trt = trt_unet(
        model_input, timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_mask,
        added_cond_kwargs={"time_ids": time_ids},
    )[0]
    torch.cuda.synchronize()
    print(f"[trt] noise_pred shape: {tuple(noise_pred_trt.shape)}")

    diff = (noise_pred_pyt.float() - noise_pred_trt.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        noise_pred_pyt.flatten().float(),
        noise_pred_trt.flatten().float(), dim=0,
    ).item()
    print(f"\n[diff]  max={diff.max().item():.4f}  mean={diff.mean().item():.6f}  "
          f"median={diff.median().item():.6f}  cos_sim={cos:.6f}")

    # Speed: warmup
    for _ in range(args.n_warmup):
        with torch.no_grad():
            _ = unet(model_input, timestep=timestep,
                     encoder_hidden_states=prompt_embeds,
                     encoder_attention_mask=prompt_mask,
                     added_cond_kwargs={"time_ids": time_ids},
                     return_dict=False)
        _ = trt_unet(model_input, timestep=timestep,
                     encoder_hidden_states=prompt_embeds,
                     encoder_attention_mask=prompt_mask,
                     added_cond_kwargs={"time_ids": time_ids})
    torch.cuda.synchronize()

    # PyTorch
    t0 = time.perf_counter()
    for _ in range(args.n_runs):
        with torch.no_grad():
            _ = unet(model_input, timestep=timestep,
                     encoder_hidden_states=prompt_embeds,
                     encoder_attention_mask=prompt_mask,
                     added_cond_kwargs={"time_ids": time_ids},
                     return_dict=False)[0]
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / args.n_runs * 1000

    # TRT
    t0 = time.perf_counter()
    for _ in range(args.n_runs):
        _ = trt_unet(model_input, timestep=timestep,
                     encoder_hidden_states=prompt_embeds,
                     encoder_attention_mask=prompt_mask,
                     added_cond_kwargs={"time_ids": time_ids})[0]
    torch.cuda.synchronize()
    trt_ms = (time.perf_counter() - t0) / args.n_runs * 1000

    print(f"\n[speed]")
    print(f"  PyTorch eager LLLite (fp16): {pt_ms:.2f} ms/call ({pt_ms/B:.2f} ms/frame)")
    print(f"  TRT LLLite-baked     (fp16): {trt_ms:.2f} ms/call ({trt_ms/B:.2f} ms/frame)")
    print(f"  speedup (UNet only):         {pt_ms / trt_ms:.2f}x")


if __name__ == "__main__":
    main()
