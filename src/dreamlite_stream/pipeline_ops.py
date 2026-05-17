"""Low-level ops over DreamLiteMobilePipeline.

Bypasses pipeline.__call__ so we can:
  - run edit at non-1024 resolutions (the official __call__ forces 1024 if image is given)
  - reuse cached prompt embeddings across frames
  - feed pre-computed image latents (skip VAE encode when prev frame is already a latent)
  - get denoise-only timing for benchmarking
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def make_time_ids(width: int, height: int, device, dtype) -> torch.Tensor:
    return torch.tensor([[width, height]], device=device, dtype=dtype)


def set_timesteps(
    pipeline,
    height: int,
    width: int,
    num_inference_steps: int,
    device,
) -> Tuple[torch.Tensor, int]:
    vae_scale = pipeline.vae_scale_factor
    lat_h, lat_w = height // vae_scale, width // vae_scale
    image_seq_len = lat_h * lat_w // 4
    cfg = pipeline.scheduler.config
    mu = _calculate_shift(
        image_seq_len,
        cfg.get("base_image_seq_len", 256),
        cfg.get("max_image_seq_len", 4096),
        cfg.get("base_shift", 0.5),
        cfg.get("max_shift", 1.16),
    )
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    pipeline.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    return pipeline.scheduler.timesteps, num_inference_steps


@torch.no_grad()
def encode_prompt_edit(
    pipeline,
    prompt: str,
    image: Image.Image,
    device,
    dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Edit-mode prompt encoding (Qwen3-VL with vision).

    NOTE: image-dependent. Re-encoding is required whenever the reference
    image changes — i.e. every frame in a streaming i2i loop. This is the
    main TE bottleneck.
    """
    decorated = (
        "[Edit]: A diptych with two side-by-side images of the same scene. "
        f"Compared to the right side, the left one has {prompt}"
    )
    return pipeline.encode_prompt(
        mode="edit",
        prompts=[decorated],
        image=image,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def encode_prompt_edit_batch(
    pipeline,
    prompts: List[str],
    images: List[Image.Image],
    device,
    dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Edit-mode TE with N independent (prompt, image) pairs in one call.

    Mirrors the inner steps of `pipeline.encode_prompt(mode="edit", ...)` but
    accepts a per-frame image list instead of one image broadcast across
    prompts. This is the path that exploits Qwen3-VL's sub-linear batch
    scaling (uncompiled, kernel-launch-bound, ~3-4x throughput at B=4).
    """
    from torch.nn.utils.rnn import pad_sequence

    if len(prompts) != len(images):
        raise ValueError(
            f"prompts and images must have the same length, got {len(prompts)} vs {len(images)}"
        )
    drop_idx = 64
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
    # BILINEAR is ~6x faster than LANCZOS for downscaling and the Q3-VL
    # vision encoder normalises away most of the high-frequency difference.
    pil_imgs = [img.resize((256, 256), Image.Resampling.BILINEAR) for img in images]

    # padding="max_length" keeps the encoder_hidden_states' sequence length
    # constant across calls regardless of prompt content. Required so a
    # later prompt change (with a different token count) doesn't force the
    # captured CUDA graph for UNet to be re-recorded on a worker thread,
    # which can hang inside cuBLAS workspace allocation during recapture.
    # 256 fits the ~64 image tokens + ~70 template tokens + room for a
    # typical prompt; smaller values truncate image_pad and crash.
    tk_out = pipeline.processor(
        text=txts, images=pil_imgs,
        padding="max_length", max_length=256, truncation=True,
        return_tensors="pt",
    ).to(device)
    outputs = pipeline.text_encoder(
        input_ids=tk_out.input_ids,
        attention_mask=tk_out.attention_mask,
        pixel_values=tk_out.pixel_values,
        image_grid_thw=tk_out.image_grid_thw,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1]

    # NOTE: the original path here was
    #     split = pipeline._extract_masked_hidden(hidden_states, tk_out.attention_mask)
    #     split = [e[drop_idx:] for e in split]
    #     prompt_embeds = pad_sequence(split, ...)
    # which forces an implicit GPU→CPU sync because variable-length per-batch
    # extraction needs to know the live mask sums. On real-time pipelines
    # that adds ~200ms of host-side wait waiting for the TE forward to drain
    # before prefetch_batch can return.
    #
    # The Q3-VL processor right-pads input_ids/attention_mask (Qwen2 tokeniser
    # default), so the system-prompt template occupies positions [0, drop_idx)
    # of every row and the user-conditioned tokens live at positions
    # [drop_idx, L_row). Slicing the padded tensor at [:, drop_idx:] is then
    # mathematically equivalent to extract-then-pad, with no sync.
    prompt_embeds = hidden_states[:, drop_idx:].to(dtype=dtype)
    prompt_embeds_mask = tk_out.attention_mask[:, drop_idx:].to(dtype=torch.long)
    return prompt_embeds, prompt_embeds_mask


@torch.no_grad()
def encode_image_to_latent_batch(
    pipeline,
    images: List[Image.Image],
    height: int,
    width: int,
    device,
    dtype,
) -> torch.Tensor:
    """VAE-encode N frames in one call. Returns latent of shape (N, C, h, w)."""
    # Skip resize when frames already match the target (the demo pipeline feeds
    # 512x512 PIL straight in, so this is the common case at B=8). Otherwise
    # use BILINEAR (LANCZOS is ~6x slower for trivial perceptual gain at VAE
    # encoder input — the encoder is robust to interpolation method).
    needs_resize = any(img.size != (width, height) for img in images)
    if needs_resize:
        resized = [img.resize((width, height), Image.Resampling.BILINEAR) for img in images]
    else:
        resized = images
    processed = pipeline.image_processor.preprocess(resized).to(device=device, dtype=dtype)
    return retrieve_latents(pipeline.vae.encode(processed), sample_mode="argmax")


@torch.no_grad()
def encode_prompt_generate(
    pipeline,
    prompt: str,
    device,
    dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate-mode prompt encoding (Qwen3-VL text-only).

    Image-independent → safe to cache across the entire video stream
    when the prompt is fixed.
    """
    decorated = f"[Generate]: {prompt}"
    return pipeline.encode_prompt(
        mode="generate",
        prompts=[decorated],
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def encode_image_to_latent(
    pipeline,
    image: Image.Image,
    height: int,
    width: int,
    device,
    dtype,
) -> torch.Tensor:
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    processed = pipeline.image_processor.preprocess(resized).to(device=device, dtype=dtype)
    return retrieve_latents(pipeline.vae.encode(processed), sample_mode="argmax")


@torch.no_grad()
def make_init_noise(
    pipeline,
    height: int,
    width: int,
    device,
    dtype,
    generator: Optional[torch.Generator] = None,
    batch_size: int = 1,
) -> torch.Tensor:
    vae_scale = pipeline.vae_scale_factor
    lat_h, lat_w = height // vae_scale, width // vae_scale
    num_channels = pipeline.vae.config.latent_channels
    shape = (batch_size, num_channels, lat_h, lat_w)
    # Generate on CPU when a generator is provided (cheap, deterministic, then ship).
    if generator is not None:
        return torch.randn(shape, generator=generator, dtype=dtype).to(device)
    return torch.randn(shape, device=device, dtype=dtype)


@torch.no_grad()
def denoise(
    pipeline,
    init_latents: torch.Tensor,
    cond_image_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    time_ids: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Run the 4-step (or N-step) flow-matching denoise loop.

    Mirrors DreamLiteMobilePipeline.__call__ inner loop. The two latents are
    concatenated along W (dim=3); UNet output is sliced back to latent width.
    """
    latents = init_latents
    for t in timesteps:
        model_input = torch.cat([latents, cond_image_latents], dim=3)
        noise_pred = pipeline.unet(
            model_input,
            timestep=t.expand(model_input.shape[0]).to(latents.dtype),
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask,
            added_cond_kwargs={"time_ids": time_ids},
            return_dict=False,
        )[0]
        noise_pred = noise_pred[..., : latents.shape[-1]]
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents


@torch.no_grad()
def decode_latent(pipeline, latents: torch.Tensor, output_type: str = "pil") -> List[Image.Image]:
    shift_factor = getattr(pipeline.vae.config, "shift_factor", 0.0)
    scaled = (latents / pipeline.vae.config.scaling_factor) + shift_factor
    decoded = pipeline.vae.decode(scaled, return_dict=False)[0]
    return pipeline.image_processor.postprocess(decoded, output_type=output_type)
