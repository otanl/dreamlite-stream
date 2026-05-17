"""Profile cond-refresh step breakdown.

The slow batches in sustained run are caused by cond-refresh; this script
times each sub-step so we know whether to optimize CPU flow, GPU CNN,
or both.
"""

from __future__ import annotations

import sys
import time
import warnings
from itertools import cycle, islice
from pathlib import Path

import cv2
import numpy as np
import torch

import torch._dynamo  # noqa: E402
torch._dynamo.config.cache_size_limit = 256

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from PIL import Image  # noqa: E402

from dreamlite import DreamLiteMobilePipeline  # noqa: E402
from dreamlite_lllite.inject import apply_lllite  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from dreamlite_stream import flow as flowlib  # noqa: E402


def main():
    print("[load] pipeline")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        str(_DREAMLITE / "models" / "DreamLite-mobile"),
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    sd = load_file(str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
    latent_hw = 512 // vae_downsample
    controller = apply_lllite(
        pipeline.unet, cond_emb_dim=32, mlp_dim=64,
        cond_image_size=512, sample_size=latent_hw,
        block_filter=["down_blocks"], inference_mode=True,
        max_batch_size=16,
    )
    controller.load_state_dict(sd, strict=False)
    controller.to(device="cuda", dtype=torch.bfloat16)
    controller.eval()
    controller.set_multiplier(1.0)

    # Load 16 frames
    cap = cv2.VideoCapture(str(_ROOT / "assets" / "davis_mp4" / "blackswan.mp4"))
    frames_np = []
    for _ in range(16):
        ok, f = cap.read()
        if not ok: break
        f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f_rgb = cv2.resize(f_rgb, (512, 512), interpolation=cv2.INTER_AREA)
        frames_np.append(f_rgb)
    cap.release()
    assert len(frames_np) == 16

    # Simulate prev_decoded = first batch's frame 0
    prev_rgb = frames_np[0]
    prev_gray = flowlib.to_gray(prev_rgb)

    # Time each step over multiple iterations
    n_iter = 5
    warmup = 1

    # 1. CPU optical flow × 16
    print(f"\n[profile] CPU optical flow x 16 frames")
    times = []
    for it in range(n_iter + warmup):
        t0 = time.perf_counter()
        for f in frames_np:
            curr_gray = flowlib.to_gray(f)
            flow = flowlib.farneback_flow(prev_gray, curr_gray)
            H, W = flow.shape[:2]
            xs, ys = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
            _ = cv2.remap(prev_rgb, xs - flow[..., 0], ys - flow[..., 1],
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if it >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    print(f"  mean: {sum(times)/len(times):.1f}ms, min: {min(times):.1f}, max: {max(times):.1f}")
    cpu_flow_ms = sum(times)/len(times)

    # Build a dummy stack for GPU steps
    stack = np.stack(frames_np, axis=0).astype(np.float32) / 255.0 * 2 - 1
    t = torch.from_numpy(stack).permute(0, 3, 1, 2).to("cuda", torch.bfloat16)

    # 2. GPU: set_cond_image (CNN forward)
    print(f"\n[profile] set_cond_image (GPU CNN x 38 hooks)")
    times = []
    for it in range(n_iter + warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        controller.set_cond_image(t)
        torch.cuda.synchronize()
        if it >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    print(f"  mean: {sum(times)/len(times):.1f}ms, min: {min(times):.1f}, max: {max(times):.1f}")
    gpu_cnn_ms = sum(times)/len(times)

    # 3. CPU + GPU end-to-end (the full cond rebuild)
    print(f"\n[profile] end-to-end cond rebuild")
    times = []
    for it in range(n_iter + warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cond_rgbs = []
        for f in frames_np:
            curr_gray = flowlib.to_gray(f)
            flow = flowlib.farneback_flow(prev_gray, curr_gray)
            H, W = flow.shape[:2]
            xs, ys = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
            warped = cv2.remap(prev_rgb, xs - flow[..., 0], ys - flow[..., 1],
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cond_rgbs.append(warped)
        stack = np.stack(cond_rgbs, axis=0).astype(np.float32) / 255.0 * 2 - 1
        tt = torch.from_numpy(stack).permute(0, 3, 1, 2).contiguous().to("cuda", torch.bfloat16)
        controller.set_cond_image(tt)
        torch.cuda.synchronize()
        if it >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    print(f"  mean: {sum(times)/len(times):.1f}ms, min: {min(times):.1f}, max: {max(times):.1f}")
    e2e_ms = sum(times)/len(times)

    print(f"\n========== summary ==========")
    print(f"CPU optical flow x 16:     {cpu_flow_ms:>6.1f}ms")
    print(f"GPU set_cond_image (CNN):  {gpu_cnn_ms:>6.1f}ms")
    print(f"end-to-end cond rebuild:   {e2e_ms:>6.1f}ms")
    print(f"\nCPU/GPU split: CPU={cpu_flow_ms/(cpu_flow_ms+gpu_cnn_ms)*100:.0f}%, GPU={gpu_cnn_ms/(cpu_flow_ms+gpu_cnn_ms)*100:.0f}%")
    if cpu_flow_ms > 2 * gpu_cnn_ms:
        print("=> CPU optical flow is the dominant cost. Threading the flow (or moving to GPU flow) is the highest-impact fix.")
    elif gpu_cnn_ms > 2 * cpu_flow_ms:
        print("=> GPU CNN is the dominant cost. Moving it to side stream (overlap with UNet) is the highest-impact fix.")
    else:
        print("=> CPU and GPU are similar. Need to optimize both for significant gain.")


if __name__ == "__main__":
    main()
