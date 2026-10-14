"""Low-level ops over DreamLiteMobilePipeline.

Bypasses pipeline.__call__ so we can:
  - run edit at non-1024 resolutions (the official __call__ forces 1024 if image is given)
  - reuse cached prompt embeddings across frames
  - feed pre-computed image latents (skip VAE encode when prev frame is already a latent)
  - get denoise-only timing for benchmarking
"""

from __future__ import annotations

import threading
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


def as_rgb_array(image) -> np.ndarray:
    """uint8 (H, W, 3) RGB array for a PIL image or an ndarray frame."""
    if isinstance(image, np.ndarray):
        return image
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image)


def as_pil(image) -> Image.Image:
    """PIL image for a PIL image or a uint8 (H, W, 3) RGB ndarray frame."""
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    return image


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


# ---------------------------------------------------------------------------
# Which hidden state is the prompt embedding
# ---------------------------------------------------------------------------
# DreamLite conditions on `hidden_states[-1]` of Qwen3-VL as transformers
# 4.57 returns it (the version upstream pins): the output of the last decoder
# layer BEFORE the language model's final norm. transformers 5 puts the
# normed tensor in that slot instead. The UNet's prompt projection ends in an
# RMSNorm, so nothing fails, but the conditioning is a different tensor (the
# projected embeddings differ by about 70% in relative terms) and the output
# changes with the installed transformers version. `prompt_hidden_state`
# returns the pre-norm tensor on either version by recording the input of
# that final norm during the forward.

_PRENORM = threading.local()


def _final_norm(text_encoder):
    lm = getattr(getattr(text_encoder, "model", None), "language_model", None)
    return getattr(lm, "norm", None)


def _record_prenorm(_module, args):
    _PRENORM.value = args[0]


def prompt_hidden_state(text_encoder, **inputs) -> torch.Tensor:
    """Run the text encoder and return the hidden state DreamLite is conditioned on.

    Per-thread, so the demo's background text-encoder thread and the pipeline
    thread do not see each other's tensors. Falls back to `hidden_states[-1]`
    for an encoder without a `model.language_model.norm`.
    """
    norm = _final_norm(text_encoder)
    if norm is not None and not getattr(norm, "_prompt_prenorm_hook", False):
        norm.register_forward_pre_hook(_record_prenorm)
        norm._prompt_prenorm_hook = True
    _PRENORM.value = None
    outputs = text_encoder(**inputs, output_hidden_states=True)
    pre = getattr(_PRENORM, "value", None)
    _PRENORM.value = None
    return pre if pre is not None else outputs.hidden_states[-1]


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
    pil_imgs = [as_pil(img).resize((256, 256), Image.Resampling.BILINEAR) for img in images]

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
    hidden_states = prompt_hidden_state(
        pipeline.text_encoder,
        input_ids=tk_out.input_ids,
        attention_mask=tk_out.attention_mask,
        pixel_values=tk_out.pixel_values,
        image_grid_thw=tk_out.image_grid_thw,
    )

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
    images = [as_pil(img) for img in images]
    needs_resize = any(img.size != (width, height) for img in images)
    if needs_resize:
        resized = [img.resize((width, height), Image.Resampling.BILINEAR) for img in images]
    else:
        resized = images
    processed = pipeline.image_processor.preprocess(resized).to(device=device, dtype=dtype)
    return retrieve_latents(pipeline.vae.encode(processed), sample_mode="argmax")


def frames_to_uint8_batch(images, height: int, width: int) -> torch.Tensor:
    """Stack frames into a CPU uint8 tensor of shape (N, H, W, 3).

    Accepts PIL images or uint8 RGB arrays. Frames already at the target
    size are stacked as they are; others are resized the same way the PIL
    path does (BILINEAR).
    """
    arrs = []
    for img in images:
        a = as_rgb_array(img)
        if a.shape[0] != height or a.shape[1] != width:
            a = np.asarray(as_pil(a).resize((width, height), Image.Resampling.BILINEAR))
        arrs.append(a)
    return torch.from_numpy(np.stack(arrs, axis=0))


@torch.no_grad()
def encode_uint8_batch_to_latent(
    vae_encoder,
    frames_u8: torch.Tensor,
    device,
    vae_dtype,
    out_dtype,
) -> torch.Tensor:
    """VAE-encode a uint8 (N, H, W, 3) batch, normalising on the GPU.

    Counterpart of `encode_image_to_latent_batch` without the host-side
    `image_processor.preprocess` (PIL -> float -> normalise), which costs
    ~5 ms per 512x512 frame on the calling thread. `vae_encoder` is the
    encoder module of an AutoencoderTiny held in `vae_dtype` /
    channels_last; the result is cast to `out_dtype` for the UNet.
    """
    x = frames_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).to(vae_dtype)
    x = x.div_(127.5).sub_(1.0).contiguous(memory_format=torch.channels_last)
    return vae_encoder(x).to(out_dtype)


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
    draw_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Init noise for `batch_size` frames.

    With a generator the draw happens on the CPU in `dtype`, or in
    `draw_dtype` and then cast when that is given. A seeded draw made
    directly in bf16 or fp16 is not the same tensor on every torch build
    (2.6 and 2.11 on Windows agree with each other; 2.12 on Linux gives an
    unrelated one), whereas the fp32 stream is the same on all of them.
    Callers that need one seed to mean one picture everywhere pass
    `draw_dtype=torch.float32`; the default keeps existing outputs as they are.
    """
    vae_scale = pipeline.vae_scale_factor
    lat_h, lat_w = height // vae_scale, width // vae_scale
    num_channels = pipeline.vae.config.latent_channels
    shape = (batch_size, num_channels, lat_h, lat_w)
    # Generate on CPU when a generator is provided (cheap, deterministic, then ship).
    if generator is not None:
        if draw_dtype is not None and draw_dtype != dtype:
            return torch.randn(shape, generator=generator, dtype=draw_dtype).to(device=device, dtype=dtype)
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


@torch.no_grad()
def decode_latent_to_uint8(pipeline, vae_decoder, latents: torch.Tensor, vae_dtype) -> np.ndarray:
    """Decode latents to a uint8 (N, H, W, 3) RGB array with one host transfer.

    Counterpart of `decode_latent` without `image_processor.postprocess`
    (float32 copy to host, then one PIL image per frame; ~4 ms per 512x512
    frame). Quantisation to uint8 happens on the GPU. `vae_decoder` is the
    decoder module of an AutoencoderTiny held in `vae_dtype` / channels_last.
    """
    shift_factor = getattr(pipeline.vae.config, "shift_factor", 0.0) or 0.0
    scaled = ((latents / pipeline.vae.config.scaling_factor) + shift_factor).to(vae_dtype)
    decoded = vae_decoder(scaled.contiguous(memory_format=torch.channels_last))
    u8 = decoded.float().clamp_(-1.0, 1.0).add_(1.0).mul_(127.5).round_().to(torch.uint8)
    return u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
