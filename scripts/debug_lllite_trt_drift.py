"""Isolate the LLLite-TRT drift source.

Tests:
 1. Zero cond_embs on both PyTorch and TRT — should match (drift independent of LLLite).
 2. Non-zero cond_embs on both — measures LLLite contribution match.
 3. PyTorch w/ LLLite vs PyTorch w/o LLLite — magnitude of LLLite delta.
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


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0,
    ).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default=str(_ROOT / "out" / "trt" / "unet_lllite_b8_512_dryrun.engine"))
    p.add_argument("--hooks_json", default=str(_ROOT / "out" / "trt" / "unet_lllite_b8_512_dryrun.hooks.json"))
    p.add_argument("--lllite_weights",
                   default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--video", default=str(_ROOT / "assets" / "davis_mp4" / "dance-twirl.mp4"))
    args = p.parse_args()
    device = torch.device("cuda")
    dtype = torch.float16
    B = 8
    SIZE = 512

    pipeline = DreamLiteMobilePipeline.from_pretrained(
        Path(_DREAMLITE / "models" / "DreamLite-mobile"), torch_dtype=dtype).to(device)
    unet = pipeline.unet
    unet.eval()
    vae_scale = pipeline.vae_scale_factor
    latent_hw = SIZE // vae_scale
    controller = apply_lllite(
        unet, cond_emb_dim=32, mlp_dim=64,
        cond_image_size=SIZE, sample_size=latent_hw,
        inference_mode=True, max_batch_size=B,
        block_filter=["down_blocks"],
    )
    sd = load_file(args.lllite_weights)
    controller.load_state_dict(sd, strict=False)
    controller.to(device=device, dtype=dtype)
    controller.eval()
    controller.set_multiplier(1.0)

    trt_unet = TRTUNetLLLiteWrapper(
        engine_path=args.engine, hooks_json=args.hooks_json,
        controller=controller, device="cuda",
    )

    cap = cv2.VideoCapture(args.video)
    pil_frames = []
    for _ in range(B):
        ok, bgr = cap.read()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        s = min(h, w); y, x = (h - s) // 2, (w - s) // 2
        rgb = cv2.resize(rgb[y:y+s, x:x+s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        pil_frames.append(Image.fromarray(rgb))
    cap.release()

    prompts = ["transfer this to oil painting style, vibrant colors"] * B
    with torch.no_grad():
        prompt_embeds, prompt_mask = ops.encode_prompt_edit_batch(
            pipeline, prompts, pil_frames, device, dtype)
        ref_latents = ops.encode_image_to_latent_batch(
            pipeline, pil_frames, SIZE, SIZE, device, dtype)
    timesteps, _ = ops.set_timesteps(pipeline, SIZE, SIZE, 1, device)
    g = torch.Generator(device="cpu").manual_seed(42)
    init_noise = torch.randn(B, 4, latent_hw, latent_hw, dtype=dtype, generator=g).to(device)
    model_input = torch.cat([init_noise, ref_latents], dim=3)
    time_ids = ops.make_time_ids(SIZE, SIZE, device, dtype).expand(B, -1)
    t = timesteps[0]
    timestep = t.expand(B).to(dtype)

    cond_np = np.stack([np.asarray(f.convert("RGB")) for f in pil_frames], axis=0).astype(np.float32)
    cond_np = cond_np / 255.0 * 2.0 - 1.0
    cond_image = torch.from_numpy(cond_np).permute(0, 3, 1, 2).contiguous().to(device, dtype)

    common = dict(
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_mask,
        added_cond_kwargs={"time_ids": time_ids},
        return_dict=False,
    )

    print("=" * 70)
    print("TEST 1: PyTorch w/ LLLite ON  vs  PyTorch w/ LLLite OFF (multiplier=0)")
    print("=" * 70)
    controller.set_cond_image(cond_image)
    controller.set_multiplier(1.0)
    with torch.no_grad():
        pyt_on = unet(model_input, **common)[0]
    controller.set_multiplier(0.0)
    with torch.no_grad():
        pyt_off = unet(model_input, **common)[0]
    print(f"  LLLite delta magnitude: cos_sim(on,off)={cos(pyt_on, pyt_off):.6f}  "
          f"max|on-off|={(pyt_on-pyt_off).abs().max().item():.4f}")

    print("\n" + "=" * 70)
    print("TEST 2: PyTorch (LLLite OFF)  vs  TRT with zero cond_embs")
    print("=" * 70)
    controller.set_multiplier(0.0)
    with torch.no_grad():
        pyt_off2 = unet(model_input, **common)[0]
    # TRT: zero cond_embs by passing zero cond_image (then they should approximate
    # m.conditioning1(0) which is NOT zero unless conditioning1 has zero bias).
    # Better: directly zero the wrapper's cond_emb cache.
    trt_unet.set_cond_image(cond_image)
    for i in range(trt_unet.n_hooks):
        trt_unet._cond_embs[i].zero_()
    trt_off = trt_unet(model_input, timestep=timestep,
                       encoder_hidden_states=prompt_embeds,
                       encoder_attention_mask=prompt_mask,
                       added_cond_kwargs={"time_ids": time_ids})[0].clone()
    torch.cuda.synchronize()
    print(f"  pyt_off2.shape={tuple(pyt_off2.shape)}  trt_off.shape={tuple(trt_off.shape)}")
    print(f"  cos_sim(pyt_off, trt_off) = {cos(pyt_off2, trt_off):.6f}")
    print(f"  max diff = {(pyt_off2-trt_off).abs().max().item():.4f}  "
          f"mean = {(pyt_off2-trt_off).abs().mean().item():.6f}")

    print("\n" + "=" * 70)
    print("TEST 3: PyTorch (LLLite ON) vs TRT (with cond_embs from set_cond_image)")
    print("=" * 70)
    controller.set_cond_image(cond_image)
    controller.set_multiplier(1.0)
    with torch.no_grad():
        pyt_on2 = unet(model_input, **common)[0]
    trt_unet.set_cond_image(cond_image)
    trt_on = trt_unet(model_input, timestep=timestep,
                      encoder_hidden_states=prompt_embeds,
                      encoder_attention_mask=prompt_mask,
                      added_cond_kwargs={"time_ids": time_ids})[0].clone()
    torch.cuda.synchronize()
    print(f"  cos_sim(pyt_on, trt_on) = {cos(pyt_on2, trt_on):.6f}")
    print(f"  max diff = {(pyt_on2-trt_on).abs().max().item():.4f}  "
          f"mean = {(pyt_on2-trt_on).abs().mean().item():.6f}")

    print("\n" + "=" * 70)
    print("TEST 4: compare per-hook cond_emb between PyTorch and TRT wrapper")
    print("=" * 70)
    # PyTorch fills m._cond_emb_buf via set_cond_image
    controller.set_cond_image(cond_image)
    for i, name in enumerate(trt_unet.hook_names[:5]):
        m = controller.modules_dict[name]
        pyt_cx = m._cond_emb_buf[:B] if m._cond_emb_buf.shape[0] >= B else m._cond_emb_buf
        trt_cx = trt_unet._cond_embs[i]
        c = cos(pyt_cx, trt_cx)
        d = (pyt_cx.float() - trt_cx.float()).abs()
        print(f"  hook[{i}] {name[:50]:50s}  cos={c:.6f}  max_diff={d.max().item():.4f}")


if __name__ == "__main__":
    main()
