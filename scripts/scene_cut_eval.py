"""Scene-cut robustness mini experiment.

Limitations §periodic cond-refresh notes that when scene content changes
abruptly (hard cut), the cached LLLite cond_emb can be up to
cond_refresh_every-1 batches stale and the adapter contribution may lag.
This script quantifies that worst case by:

  1. Building synthetic hard-cut clips that concatenate two DAVIS-2017
     sequences with no transition.
  2. Running the champion config on each clip at:
       - cond_refresh_every = 8 (default; stale embedding window)
       - cond_refresh_every = 1 (always-refresh oracle; perfect cut handling)
  3. Computing warping_error on:
       - the pre-cut window  (intra first clip; control)
       - the post-cut window (recovery; where staleness hurts)

The N=1 column upper-bounds what a perfect cut-triggered refresh
trigger could recover; the gap between N=8 and N=1 in the post-cut
window is the cost a deployment pays at the worst possible cut timing.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import warnings
from pathlib import Path
from statistics import mean, stdev

import cv2
import numpy as np
import torch
import torch._dynamo

torch._dynamo.config.cache_size_limit = 256

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
_LLLITE = _ROOT.parent / "dreamlite-lllite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))
sys.path.insert(0, str(_LLLITE / "src"))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402
from dreamlite_lllite import apply_lllite  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from dreamlite_stream import BatchedEditWorker, SharedState  # noqa: E402
from dreamlite_stream.metrics import (  # noqa: E402
    compute_temporal, read_video_frames,
)
from dreamlite_stream.runtime import VideoWriter, iter_video_frames  # noqa: E402

DAVIS_DIR = _ROOT / "assets" / "davis_mp4"
CUT_DIR = _ROOT / "assets" / "scene_cut"
OUT_ROOT = _ROOT / "out" / "scene_cut"

# (name, clipA, clipB)  -- first 32 frames of A then first 32 of B; cut at 32.
CUTS = [
    ("blackswan_goat", "blackswan", "goat"),
    ("kite_dance", "kite-surf", "dance-twirl"),
    ("libby_camel", "libby", "camel"),
]
PRE_LEN = 32
POST_LEN = 32
CUT_AT = PRE_LEN  # frame index where the cut occurs


def build_cut_clip(name: str, a: str, b: str, size: int = 512) -> Path:
    """Concatenate first PRE_LEN frames of a with first POST_LEN of b at
    size x size center-crop+resize. Output is fps=24 .mp4."""
    out = CUT_DIR / f"{name}.mp4"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    src_a = DAVIS_DIR / f"{a}.mp4"
    src_b = DAVIS_DIR / f"{b}.mp4"
    frames = []
    for src, n_take in [(src_a, PRE_LEN), (src_b, POST_LEN)]:
        cap = cv2.VideoCapture(str(src))
        taken = 0
        while taken < n_take:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w, _ = bgr.shape
            s = min(h, w)
            y0, x0 = (h - s) // 2, (w - s) // 2
            bgr = bgr[y0:y0+s, x0:x0+s]
            bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(bgr)
            taken += 1
        cap.release()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out), fourcc, 24.0, (size, size))
    for f in frames:
        vw.write(f)
    vw.release()
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights",
                   default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=PRE_LEN + POST_LEN)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--n_refresh", type=int, nargs="+", default=[8, 1])
    return p.parse_args()


def windowed_warp_err(input_frames, output_frames, start, end):
    sl_in = input_frames[start:end]
    sl_out = output_frames[start:end]
    n = min(len(sl_in), len(sl_out))
    if n < 2:
        return float("nan")
    return compute_temporal(sl_in[:n], sl_out[:n]).warping_error


@torch.no_grad()
def run_one(worker, args, in_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter_video_frames(str(in_path), args.size)
    writer = None
    fps_global = 24.0
    n_total = 0
    cur_buf = []

    def collect_batch(it):
        nonlocal n_total, fps_global
        buf = []
        for idx, frame, fps in it:
            if idx >= args.max_frames:
                return buf, False
            fps_global = fps
            buf.append(frame)
            n_total += 1
            if len(buf) >= args.batch_size:
                return buf, True
        return buf, False

    cur_buf, more = collect_batch(iterator)
    if not cur_buf or len(cur_buf) != args.batch_size:
        return 0
    cur_pf = worker.prefetch_batch(cur_buf)
    while True:
        if more:
            nxt_buf, more = collect_batch(iterator)
            if len(nxt_buf) < args.batch_size:
                nxt_buf = []
        else:
            nxt_buf = []
        nxt_pf = worker.prefetch_batch(nxt_buf) if nxt_buf else None
        outputs, t = worker.step_batch_with_prefetch(cur_buf, cur_pf)
        if writer is None:
            writer = VideoWriter(str(out_path), args.size, fps_global)
        for img in outputs:
            writer.write_pil(img)
        if not nxt_buf:
            break
        cur_buf, cur_pf = nxt_buf, nxt_pf
    if writer:
        writer.close()
    return n_total


def main():
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
    latent_hw = args.size // vae_downsample
    block_filter = [s.strip() for s in args.lllite_blocks.split(",")] if args.lllite_blocks else None
    controller = apply_lllite(
        pipeline.unet, cond_emb_dim=32, mlp_dim=64,
        cond_image_size=args.size, sample_size=latent_hw,
        inference_mode=True, max_batch_size=args.batch_size,
        block_filter=block_filter,
    )
    sd = load_file(args.lllite_weights)
    controller.load_state_dict(sd, strict=False)
    controller.to(device=args.device, dtype=torch.bfloat16)
    controller.eval()
    controller.set_multiplier(1.0)
    print(f"[lllite] {len(controller.modules_dict)} hooks attached ({args.lllite_blocks})")

    # Build cuts up front
    cut_paths = []
    for name, a, b in CUTS:
        p = build_cut_clip(name, a, b, args.size)
        cut_paths.append((name, p))
        print(f"[cut] {name}: {p}")

    # We need to instantiate a worker per refresh setting because
    # cond_refresh_every is a static dataclass field; rebuilding the
    # worker also clears the cond cache between configs cleanly.
    results = []
    for n_refresh in args.n_refresh:
        print(f"\n=== cond_refresh_every = {n_refresh} ===")
        state = SharedState(
            height=args.size, width=args.size,
            num_inference_steps=args.steps, prompt=args.prompt,
        )
        worker = BatchedEditWorker(
            pipeline=pipeline, state=state,
            batch_size=args.batch_size, device=args.device,
            dtype=torch.bfloat16, seed=args.seed,
            compile=False,
            lllite_controller=controller,
            cond_refresh_every=n_refresh,
            cond_flow_workers=8,
        )

        for name, in_path in cut_paths:
            out_path = OUT_ROOT / f"N{n_refresh}" / f"{name}.mp4"
            worker.state.frame_idx = 0
            worker._last_prev_decoded = None
            worker._last_prev_input_gray = None
            worker._cond_call_idx = 0
            gc.collect()
            torch.cuda.empty_cache()

            n = run_one(worker, args, in_path, out_path)
            if n < PRE_LEN + POST_LEN:
                print(f"  {name}: only {n} frames produced; skip")
                continue
            in_frames = read_video_frames(str(in_path), size=args.size)[:PRE_LEN + POST_LEN]
            out_frames = read_video_frames(str(out_path), size=args.size)[:PRE_LEN + POST_LEN]
            we_pre = windowed_warp_err(in_frames, out_frames, 0, CUT_AT)
            we_post8 = windowed_warp_err(in_frames, out_frames, CUT_AT, CUT_AT + 8)
            we_post16 = windowed_warp_err(in_frames, out_frames, CUT_AT, CUT_AT + 16)
            we_full = windowed_warp_err(in_frames, out_frames, 0, PRE_LEN + POST_LEN)
            row = {
                "n_refresh": n_refresh, "clip": name,
                "warp_err_pre": we_pre,
                "warp_err_post8": we_post8,
                "warp_err_post16": we_post16,
                "warp_err_full": we_full,
            }
            results.append(row)
            print(f"  {name:18s} pre={we_pre:6.2f}  post8={we_post8:6.2f}  post16={we_post16:6.2f}  full={we_full:6.2f}")

        del worker
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate
    print("\n=== Aggregate (mean +- std over 3 clips) ===")
    print(f"{'N':>3s}  {'pre':>13s}  {'post8':>13s}  {'post16':>13s}  {'full':>13s}")
    for n_refresh in args.n_refresh:
        rs = [r for r in results if r["n_refresh"] == n_refresh]
        if not rs:
            continue
        pre = [r["warp_err_pre"] for r in rs]
        p8 = [r["warp_err_post8"] for r in rs]
        p16 = [r["warp_err_post16"] for r in rs]
        full = [r["warp_err_full"] for r in rs]
        def fmt(v):
            return f"{mean(v):5.2f}+-{stdev(v) if len(v)>1 else 0:.2f}"
        print(f"  {n_refresh:>3d}  {fmt(pre):>13s}  {fmt(p8):>13s}  {fmt(p16):>13s}  {fmt(full):>13s}")

    out_jsonl = OUT_ROOT / "results.jsonl"
    with open(out_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[saved] {out_jsonl}")


if __name__ == "__main__":
    main()
