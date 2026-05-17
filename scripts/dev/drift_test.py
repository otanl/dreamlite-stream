"""Long-sequence drift test: loop a DAVIS clip and measure metric drift
across chunks. Defends the claim "stable over multi-minute clips".

For T total frames split into N chunks, computes per-chunk warp_err,
Sobel, HF-FFT, LPIPS-to-4-step-ref. Drift = late chunks vs early chunks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import cycle, islice
from pathlib import Path
from statistics import mean, stdev

import cv2
import lpips
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from PIL import Image  # noqa: E402

from dreamlite import DreamLiteMobilePipeline  # noqa: E402
from dreamlite_lllite.inject import apply_lllite  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from dreamlite_stream import BatchedEditWorker, SharedState  # noqa: E402
from dreamlite_stream.metrics import compute_temporal, hf_density  # noqa: E402
from dreamlite_stream.runtime import VideoWriter  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights", default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--mp4", default=str(_ROOT / "assets" / "davis_mp4" / "parkour.mp4"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--total_frames", type=int, default=480)
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--cond_refresh_every", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup_batches", type=int, default=2)
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--out_dir", default=str(_ROOT / "out" / "drift_test"))
    return p.parse_args()


def load_and_loop_frames(mp4: str, total: int, size: int):
    """Read mp4, return PIL frames looped to `total` length."""
    cap = cv2.VideoCapture(mp4)
    raw = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f_rgb = cv2.resize(f_rgb, (size, size), interpolation=cv2.INTER_AREA)
        raw.append(Image.fromarray(f_rgb))
    cap.release()
    if not raw:
        raise RuntimeError(f"empty mp4 {mp4}")
    # Loop cycles
    looped = list(islice(cycle(raw), total))
    return looped


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    # LLLite (down_blocks subset, champion configuration)
    print(f"[lllite] {args.lllite_weights}")
    sd = load_file(args.lllite_weights)
    vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
    latent_hw = args.size // vae_downsample
    controller = apply_lllite(
        pipeline.unet, cond_emb_dim=32, mlp_dim=64,
        cond_image_size=args.size, sample_size=latent_hw,
        block_filter=[args.lllite_blocks], inference_mode=True,
        max_batch_size=args.batch_size,
    )
    controller.load_state_dict(sd, strict=False)
    controller.to(device=args.device, dtype=torch.bfloat16)
    controller.eval()
    controller.set_multiplier(1.0)

    # Frames
    frames = load_and_loop_frames(args.mp4, args.total_frames, args.size)
    print(f"[frames] {len(frames)} frames (looped)")

    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=args.steps, prompt=args.prompt,
    )
    worker = BatchedEditWorker(
        pipeline=pipeline, state=state, batch_size=args.batch_size,
        device=args.device, dtype=torch.bfloat16, seed=args.seed,
        compile=True, compile_mode="reduce-overhead",
        lllite_controller=controller,
        cond_refresh_every=args.cond_refresh_every,
    )

    out_path = out_dir / f"drift_{Path(args.mp4).stem}_{args.total_frames}f.mp4"
    writer = VideoWriter(str(out_path), args.size, 24.0)
    all_out_frames = []
    all_in_frames = []
    batch_timings = []
    batch_idx = 0

    print(f"[run] batches of {args.batch_size}, refresh every {args.cond_refresh_every}")
    cur_buf = None
    cur_pf = None
    n_batches = len(frames) // args.batch_size
    for b in range(n_batches):
        buf = frames[b * args.batch_size : (b + 1) * args.batch_size]
        if cur_buf is None:
            cur_buf = buf
            cur_pf = worker.prefetch_batch(cur_buf)
            continue
        nxt_pf = worker.prefetch_batch(buf)
        t0 = time.perf_counter()
        out_pils, _ = worker.step_batch_with_prefetch(cur_buf, cur_pf)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        for pil in out_pils:
            writer.write_pil(pil)
        if batch_idx >= args.warmup_batches:
            batch_timings.append(dt)
            for pil, fr in zip(out_pils, cur_buf):
                all_out_frames.append(np.asarray(pil.convert("RGB")))
                all_in_frames.append(np.asarray(fr.convert("RGB")))
        batch_idx += 1
        cur_buf, cur_pf = buf, nxt_pf
    # Process the final batch
    out_pils, _ = worker.step_batch_with_prefetch(cur_buf, cur_pf)
    for pil in out_pils:
        writer.write_pil(pil)
    if batch_idx >= args.warmup_batches:
        for pil, fr in zip(out_pils, cur_buf):
            all_out_frames.append(np.asarray(pil.convert("RGB")))
            all_in_frames.append(np.asarray(fr.convert("RGB")))
    writer.close()

    print(f"\n[done] {len(all_out_frames)} measured frames")
    print(f"avg fps: {(len(batch_timings) * args.batch_size) / sum(batch_timings):.2f}")

    # Chunked drift analysis
    print(f"\n[drift] chunking by {args.chunk_size} frames")
    print(f"{'chunk':<8} {'frames':>10} {'warp':>8} {'sobel':>8} {'hf':>8} {'lpips':>8}")
    lpips_net = lpips.LPIPS(net="alex", verbose=False).to(args.device).eval()
    # First chunk's avg output frame as reference for LPIPS drift
    n = len(all_out_frames)
    chunks = []
    for start in range(0, n - args.chunk_size + 1, args.chunk_size):
        end = start + args.chunk_size
        in_ck = all_in_frames[start:end]
        out_ck = all_out_frames[start:end]
        m = compute_temporal(in_ck, out_ck)
        sob, hf = hf_density(out_ck)
        # LPIPS: each frame vs same-position frame in first chunk (loop fairness)
        lp_vals = []
        for i in range(len(out_ck)):
            ref_idx = i % args.chunk_size  # same phase in first chunk
            if ref_idx >= len(all_out_frames[:args.chunk_size]):
                continue
            a = all_out_frames[:args.chunk_size][ref_idx]
            b = out_ck[i]
            ta = torch.from_numpy(a).to(args.device).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            tb = torch.from_numpy(b).to(args.device).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            lp_vals.append(lpips_net(ta, tb).item())
        lp = mean(lp_vals) if lp_vals else 0.0
        chunks.append({
            "start": start, "end": end,
            "warp": m.warping_error, "sobel": sob, "hf_fft": hf, "lpips": lp,
        })
        print(f"{start//args.chunk_size:<8} {start}-{end:<9} {m.warping_error:>8.2f} {sob:>8.2f} {hf:>8.0f} {lp:>8.3f}")

    # Save
    out_jsonl = out_dir / f"drift_{Path(args.mp4).stem}_{args.total_frames}f.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"\n[saved] {out_path}")
    print(f"[saved] {out_jsonl}")


if __name__ == "__main__":
    main()
