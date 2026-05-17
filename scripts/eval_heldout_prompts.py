"""Eval v4 multi-prompt LLLite on prompts NOT seen during training.

The v4 LLLite was distilled on 5 styles (oil/watercolor/pencil/anime/3d).
This script evaluates the same LLLite on 3 held-out styles to test
prompt generalization beyond the training prompt set.
"""

from __future__ import annotations

import argparse
import json
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

from dreamlite_stream import BatchedEditWorker, SharedState  # noqa: E402
from dreamlite_stream.metrics import compute_temporal, hf_density  # noqa: E402
from dreamlite_stream.runtime import VideoWriter, iter_video_frames  # noqa: E402


# Held-out prompts (NOT in v4 training set)
HELDOUT_PROMPTS = [
    ("comic",   "transfer this to comic book style, halftone shading, bold outlines"),
    ("ukiyoe",  "transfer this to ukiyo-e woodblock print style, flat colors, line art"),
    ("vangogh", "transfer this to van gogh impressionist style, swirling brushstrokes"),
]

SEQUENCES = [
    "blackswan", "libby", "swing", "camel", "dance-twirl",
    "goat", "bmx-trees", "parkour", "kite-surf",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights", default=str(_ROOT / "runs" / "temporal_lllite_v4_multiprompt" / "temporal_lllite_step001644.safetensors"))
    p.add_argument("--mp4_dir", default=str(_ROOT / "assets" / "davis_mp4"))
    p.add_argument("--sequences", nargs="+", default=SEQUENCES)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--cond_refresh_every", type=int, default=8)
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=48)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out_dir", default=str(_ROOT / "out" / "heldout_prompts_eval"))
    return p.parse_args()


@torch.no_grad()
def run_sequence(worker, state, args, seq_name: str, prompt_tag: str):
    in_path = Path(args.mp4_dir) / f"{seq_name}.mp4"
    if not in_path.exists():
        return None
    state.reset()
    worker._cond_call_idx = 0

    timings = []
    out_frames = []
    in_frames = []

    def collect_batch(it):
        buf = []
        for idx, frame, _ in it:
            if idx >= args.max_frames:
                return buf, False
            buf.append(frame)
            if len(buf) >= args.batch_size:
                return buf, True
        return buf, False

    iterator = iter_video_frames(str(in_path), args.size)
    cur_buf, more = collect_batch(iterator)
    if not (cur_buf and len(cur_buf) == args.batch_size):
        return None
    out_path = Path(args.out_dir) / f"{prompt_tag}_{seq_name}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoWriter(str(out_path), args.size, 24.0)
    cur_pf = worker.prefetch_batch(cur_buf)
    batch_idx = 0
    while True:
        if more:
            nxt_buf, more = collect_batch(iterator)
            if len(nxt_buf) < args.batch_size:
                nxt_buf = []
        else:
            nxt_buf = []
        nxt_pf = worker.prefetch_batch(nxt_buf) if nxt_buf else None
        t0 = time.perf_counter()
        out_pils, _ = worker.step_batch_with_prefetch(cur_buf, cur_pf)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        for pil in out_pils:
            writer.write_pil(pil)
        if batch_idx >= args.warmup:
            timings.append(dt)
            for pil, fr in zip(out_pils, cur_buf):
                out_frames.append(np.asarray(pil.convert("RGB")))
                in_frames.append(np.asarray(fr.convert("RGB")) if hasattr(fr, "convert") else np.asarray(fr))
        batch_idx += 1
        if not nxt_buf:
            break
        cur_buf, cur_pf = nxt_buf, nxt_pf
    writer.close()

    if not timings:
        return None
    fps = (len(timings) * args.batch_size) / sum(timings)
    metrics = compute_temporal(in_frames, out_frames)
    sobel, hf = hf_density(out_frames)
    return {
        "sequence": seq_name, "prompt_tag": prompt_tag,
        "fps": float(fps),
        "warp_err": float(metrics.warping_error),
        "sobel": float(sobel), "hf_fft": float(hf),
        "n_frames": len(out_frames),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    print(f"[setup] {len(HELDOUT_PROMPTS)} held-out prompts x {len(args.sequences)} sequences")

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    print(f"[lllite] {args.lllite_weights}")
    sd = load_file(args.lllite_weights)
    vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
    latent_hw = args.size // vae_downsample
    blocks = [args.lllite_blocks] if isinstance(args.lllite_blocks, str) else args.lllite_blocks
    controller = apply_lllite(
        pipeline.unet, cond_emb_dim=32, mlp_dim=64,
        cond_image_size=args.size, sample_size=latent_hw,
        block_filter=blocks, inference_mode=True, max_batch_size=args.batch_size,
    )
    controller.load_state_dict(sd, strict=False)
    controller.to(device=args.device, dtype=torch.bfloat16)
    controller.eval()
    controller.set_multiplier(1.0)

    # Single worker reused across (prompt, seq) — avoid Dynamo cache cascade
    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=args.steps, prompt=HELDOUT_PROMPTS[0][1],
    )
    worker = BatchedEditWorker(
        pipeline=pipeline, state=state, batch_size=args.batch_size,
        device=args.device, dtype=torch.bfloat16, seed=args.seed,
        compile=True, compile_mode="reduce-overhead",
        lllite_controller=controller,
        cond_refresh_every=args.cond_refresh_every,
    )

    all_results = []
    with results_path.open("w", encoding="utf-8") as f:
        for tag, prompt in HELDOUT_PROMPTS:
            print(f"\n========== held-out {tag}: {prompt!r} ==========")
            state.prompt = prompt
            for seq in args.sequences:
                print(f"  [{tag}/{seq}] running...")
                r = run_sequence(worker, state, args, seq, tag)
                if r is None:
                    print(f"    [skip] {seq}")
                    continue
                f.write(json.dumps(r) + "\n")
                f.flush()
                all_results.append(r)
                print(f"    fps={r['fps']:.2f}  warp={r['warp_err']:.2f}  sobel={r['sobel']:.2f}  hf={r['hf_fft']:.0f}")

    print("\n========== aggregate ==========")
    print(f"{'prompt':<10}{'n':>3}  {'fps':>14}  {'warp':>14}  {'sobel':>14}  {'hf':>14}")
    for tag, _ in HELDOUT_PROMPTS:
        rs = [r for r in all_results if r["prompt_tag"] == tag]
        if not rs:
            continue
        fps = [r["fps"] for r in rs]
        we = [r["warp_err"] for r in rs]
        sb = [r["sobel"] for r in rs]
        hf = [r["hf_fft"] for r in rs]
        n = len(fps)
        sd = lambda xs: stdev(xs) if n >= 2 else 0.0
        print(f"{tag:<10}{n:>3}  {mean(fps):>6.2f}+-{sd(fps):>4.2f}  "
              f"{mean(we):>6.2f}+-{sd(we):>4.2f}  "
              f"{mean(sb):>6.2f}+-{sd(sb):>4.2f}  "
              f"{mean(hf):>6.0f}+-{sd(hf):>4.0f}")
    print(f"\n[saved] {results_path}")


if __name__ == "__main__":
    main()
