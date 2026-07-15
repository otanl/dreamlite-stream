"""Streaming video stylization on Mobile-O: three-mode comparison.

first actual video-to-video run on the Mobile-O substrate.

Background (substrate-axis finding): on DreamLite, frame content enters
through the UNet img2img path every frame while the TE cond carries the
style instruction, so R1 cond-caching leaves video tracking intact. On
Mobile-O's edit path, frame content enters ONLY through the conditioning
(frame -> MobileCLIP -> LLM -> MCP -> encoder_hidden_states) and the
denoise starts from pure noise -- caching the cond freezes the output.
The adaptation is SDEdit-style img2img: encode the current frame with
the SANA VAE, noise it to an intermediate timestep, and denoise the
remaining steps with the (cacheable) instruction cond. Content then
flows per frame through the latent path and R1 applies as designed.

Modes:
  naive   -- full TE every frame, pure-noise generation (Mobile-O as-is)
  frozen  -- R1 cond cache + pure-noise generation (demonstrates freeze)
  sdedit  -- SDEdit img2img + R1 cond cache (adapted recipe)

Run (from F:/work/Mobile-O):
    .venv-mobileo/Scripts/python.exe mobile_o_video.py \
        --sequence libby --frames 24 --refresh-every 8 \
        --steps 8 --strength 0.6
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mobileo.model.builder import load_pretrained_model  # noqa: E402
from diffusers.utils.torch_utils import randn_tensor  # noqa: E402
from streaming_recipe import build_edit_inputs, te_forward  # noqa: E402

DAVIS_ROOT = r"F:\work\dreamlite-stream\assets\davis\DAVIS\JPEGImages\480p"


# ---------------------------------------------------------------------------
# Frame I/O
# ---------------------------------------------------------------------------


def load_davis_frames(sequence: str, n: int, size: int = 512):
    from PIL import Image
    seq_dir = os.path.join(DAVIS_ROOT, sequence)
    jpgs = sorted(os.listdir(seq_dir))[:n]
    frames = []
    for j in jpgs:
        im = Image.open(os.path.join(seq_dir, j)).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2,
                      (w - s) // 2 + s, (h - s) // 2 + s))
        frames.append(im.resize((size, size), Image.BICUBIC))
    return frames


def frames_to_mp4(frame_dir: str, out_mp4: str, fps: int = 12):
    """Assemble saved PNGs into an mp4 via av (already in the venv)."""
    import av
    from PIL import Image
    pngs = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
    if not pngs:
        return False
    first = Image.open(os.path.join(frame_dir, pngs[0]))
    container = av.open(out_mp4, "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = first.size
    stream.pix_fmt = "yuv420p"
    for p in pngs:
        im = Image.open(os.path.join(frame_dir, p)).convert("RGB")
        frame = av.VideoFrame.from_image(im)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return True


# ---------------------------------------------------------------------------
# SDEdit img2img denoise (content via latents, instruction via cond)
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_frame(model, pil_image):
    """Frame -> SANA VAE latents (scaled)."""
    import numpy as np
    vae = model.get_model().get_sana_vae()
    arr = np.asarray(pil_image).astype("float32") / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    x = (x * 2.0 - 1.0).to(vae.device, dtype=vae.dtype)
    latent = vae.encode(x).latent
    return latent * vae.config.scaling_factor


@torch.no_grad()
def sdedit_denoise(model, cond, frame_latents, num_inference_steps=8,
                   strength=0.6, guidance_scale=1.5, with_cfg=True,
                   generator=None):
    """Noise the frame latents to an intermediate timestep, denoise the rest.

    strength in (0, 1]: fraction of the schedule actually run.
    1.0 == full generation from noise; smaller == closer to the input.
    """
    sched = model.model.noise_scheduler
    device = frame_latents.device
    sched.set_timesteps(num_inference_steps)
    timesteps = sched.timesteps
    t_start = max(num_inference_steps - int(num_inference_steps * strength), 0)
    run_timesteps = timesteps[t_start:]

    noise = randn_tensor(frame_latents.shape, generator=generator,
                         device=device, dtype=torch.float32)
    latents = sched.add_noise(frame_latents.float(), noise,
                              run_timesteps[:1])

    for t in run_timesteps:
        latent_in = torch.cat([latents] * 2) if with_cfg else latents
        if hasattr(sched, "scale_model_input"):
            latent_in = sched.scale_model_input(latent_in, t)
        noise_pred = model.get_model().dit(
            hidden_states=latent_in.to(torch.bfloat16),
            encoder_hidden_states=cond.to(torch.bfloat16),
            timestep=t.unsqueeze(0).expand(latent_in.shape[0]).to(device),
            encoder_attention_mask=None,
        ).sample.float()
        if with_cfg:
            uncond, text = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (text - uncond)
        latents = sched.step(noise_pred, t, latents).prev_sample
    return model.decode_latents(latents.to(model.model.vae.dtype))


@torch.no_grad()
def noise_denoise(model, cond, latent_shape, num_inference_steps=8,
                  guidance_scale=1.5, with_cfg=True, generator=None):
    """Pure-noise generation (modes naive / frozen)."""
    sched = model.model.noise_scheduler
    device = cond.device
    latents = randn_tensor(latent_shape, generator=generator,
                           device=device, dtype=torch.float32)
    sched.set_timesteps(num_inference_steps)
    for t in sched.timesteps:
        latent_in = torch.cat([latents] * 2) if with_cfg else latents
        if hasattr(sched, "scale_model_input"):
            latent_in = sched.scale_model_input(latent_in, t)
        noise_pred = model.get_model().dit(
            hidden_states=latent_in.to(torch.bfloat16),
            encoder_hidden_states=cond.to(torch.bfloat16),
            timestep=t.unsqueeze(0).expand(latent_in.shape[0]).to(device),
            encoder_attention_mask=None,
        ).sample.float()
        if with_cfg:
            uncond, text = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (text - uncond)
        latents = sched.step(noise_pred, t, latents).prev_sample
    return model.decode_latents(latents.to(model.model.vae.dtype))


# ---------------------------------------------------------------------------
# Temporal stability proxy (informal, RAFT-free)
# ---------------------------------------------------------------------------


def temporal_stats(frames_pil):
    """Mean abs diff between consecutive frames, in [0,255] units."""
    import numpy as np
    if len(frames_pil) < 2:
        return 0.0
    diffs = []
    prev = np.asarray(frames_pil[0]).astype("float32")
    for f in frames_pil[1:]:
        cur = np.asarray(f).astype("float32")
        diffs.append(np.abs(cur - prev).mean())
        prev = cur
    return float(sum(diffs) / len(diffs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="checkpoints/Mobile-O-0.5B")
    ap.add_argument("--sequence", default="libby")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--refresh-every", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--instruction",
                    default="make it look like an oil painting")
    ap.add_argument("--modes", nargs="+",
                    default=["naive", "frozen", "sdedit"])
    ap.add_argument("--out-root", default="smoke_outputs/video")
    args = ap.parse_args()

    print("loading model ...", flush=True)
    tokenizer, model, _ = load_pretrained_model(args.model_path)
    model.to("cuda:0")
    model.to(torch.bfloat16)

    frames = load_davis_frames(args.sequence, args.frames)
    print(f"sequence={args.sequence} frames={len(frames)}", flush=True)
    src_stat = temporal_stats(frames)
    print(f"source temporal mean-abs-diff: {src_stat:.2f}", flush=True)

    dit = model.get_model().dit
    latent_shape = (1, dit.config.in_channels,
                    dit.config.sample_size, dit.config.sample_size)

    summary = {}
    for mode in args.modes:
        out_dir = os.path.join(args.out_root,
                               f"{args.sequence}_{mode}")
        os.makedirs(out_dir, exist_ok=True)
        cached_cond = None
        n_te = 0
        edited = []
        gen = torch.Generator(device="cuda").manual_seed(42)
        fixed_noise_gen = torch.Generator(device="cuda").manual_seed(42)
        torch.cuda.synchronize()
        t0 = time.time()
        for i, frame in enumerate(frames):
            need_te = (mode == "naive") or (cached_cond is None) or \
                      (i % args.refresh_every == 0)
            if need_te:
                input_ids, image_tensor = build_edit_inputs(
                    tokenizer, model, frame, args.instruction)
                cached_cond = te_forward(model, input_ids, image_tensor)
                n_te += 1
            if mode in ("sdedit", "sdedit_aligned"):
                lat = encode_frame(model, frame)
                if mode == "sdedit_aligned":
                    # Cross-frame noise alignment (StreamDiffusion-style):
                    # the SAME noise tensor perturbs every frame's latents,
                    # so frame-to-frame output differences come from the
                    # content, not from independent noise draws.
                    gen.manual_seed(42)
                img = sdedit_denoise(model, cached_cond, lat,
                                     num_inference_steps=args.steps,
                                     strength=args.strength,
                                     generator=gen)[0]
            else:
                # fixed noise per frame so 'frozen' visibly freezes rather
                # than flickering randomly
                fixed_noise_gen.manual_seed(42)
                img = noise_denoise(model, cached_cond, latent_shape,
                                    num_inference_steps=args.steps,
                                    generator=fixed_noise_gen)[0]
            img.save(os.path.join(out_dir, f"{i:05d}.png"))
            edited.append(img)
        torch.cuda.synchronize()
        wall = time.time() - t0
        stat = temporal_stats(edited)
        mp4 = os.path.join(args.out_root,
                           f"{args.sequence}_{mode}.mp4")
        ok = frames_to_mp4(out_dir, mp4)
        summary[mode] = (n_te, wall / len(frames), stat, mp4 if ok else "-")
        print(f"[{mode}] TE calls={n_te}, {wall/len(frames)*1e3:.0f} ms/frame "
              f"(informal), temporal mean-abs-diff={stat:.2f}, mp4={mp4}",
              flush=True)

    print("\n=== summary (informal, shared GPU) ===", flush=True)
    print(f"  source:  temporal-diff {src_stat:.2f}  (tracking target)",
          flush=True)
    for mode, (n_te, spf, stat, mp4) in summary.items():
        print(f"  {mode:7s} TE={n_te:3d}  {spf*1e3:6.0f} ms/f  "
              f"temporal-diff {stat:6.2f}  {mp4}", flush=True)
    print("\nInterpretation: 'frozen' should show temporal-diff ~0 within "
          "refresh windows (content frozen by cond cache); 'sdedit' should "
          "track the source statistic while keeping TE calls at "
          "ceil(N/refresh_every).", flush=True)
    print("MOBILE_O_VIDEO_DONE", flush=True)


if __name__ == "__main__":
    main()
