"""Held-out video eval: v5-heldout LLLite (trained on 7 DAVIS sequences)
evaluated on the 3 held-out DAVIS sequences. Compare to v3 LLLite
(trained on all 10) on the same 3 sequences for in-domain reference.

Outputs mp4 + Sobel/HF/warp metrics.
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


HELDOUT_SEQUENCES = ["bmx-trees", "parkour", "kite-surf"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights", required=True)
    p.add_argument("--lllite_label", default="v5-heldout")
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--mp4_dir", default=str(_ROOT / "assets" / "davis_mp4"))
    p.add_argument("--sequences", nargs="+", default=HELDOUT_SEQUENCES)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--cond_refresh_every", type=int, default=8)
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=48)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out_dir", default=str(_ROOT / "out" / "heldout_video_eval"))
    return p.parse_args()


@torch.no_grad()
def run_sequence(worker, state, args, seq_name: str, label: str):
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
    out_path = Path(args.out_dir) / f"{label}_{seq_name}.mp4"
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
        "sequence": seq_name, "label": label,
        "fps": float(fps),
        "warp_err": float(metrics.warping_error),
        "sobel": float(sobel), "hf_fft": float(hf),
        "n_frames": len(out_frames),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"results_{args.lllite_label}.jsonl"
    if results_path.exists():
        results_path.unlink()

    print(f"[setup] LLLite={args.lllite_label}, sequences={args.sequences}")

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

    rows = []
    with results_path.open("w", encoding="utf-8") as f:
        for seq in args.sequences:
            print(f"  [{args.lllite_label}/{seq}] running...")
            r = run_sequence(worker, state, args, seq, args.lllite_label)
            if r is None:
                print(f"    [skip] {seq}")
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            rows.append(r)
            print(f"    fps={r['fps']:.2f}  warp={r['warp_err']:.2f}  sobel={r['sobel']:.2f}  hf={r['hf_fft']:.0f}")

    if rows:
        fps = [r["fps"] for r in rows]
        we = [r["warp_err"] for r in rows]
        sb = [r["sobel"] for r in rows]
        hf = [r["hf_fft"] for r in rows]
        n = len(fps)
        sd = lambda xs: stdev(xs) if n >= 2 else 0.0
        print(f"\n[{args.lllite_label}] aggregate n={n}: "
              f"fps={mean(fps):.2f}+-{sd(fps):.2f}  "
              f"warp={mean(we):.2f}+-{sd(we):.2f}  "
              f"sobel={mean(sb):.2f}+-{sd(sb):.2f}  "
              f"hf={mean(hf):.0f}+-{sd(hf):.0f}")
    print(f"[saved] {results_path}")


if __name__ == "__main__":
    main()
