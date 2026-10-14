"""Run the champion config on every DAVIS sequence and report aggregate
speed and quality.

Champion config (from v3 sweep):
    batched B=8 + 1-step + compile + pipelined
    LLLite v3 with --lllite_blocks down_blocks  (38 hooks)
    --cond_refresh_every 8

Pass --lean (or the individual flags it stands for) to run the reduced
per-batch path instead: prompt embedding and LLLite cond computed once per
clip, one shared init-noise draw, and the fp16 compiled VAE with uint8 I/O.
See dreamlite_stream/lean.py. The defaults reproduce the paper's numbers.

Reports per-sequence:
    fps        step-only: frames / time inside step_batch_with_prefetch
    fps_loop   prefetch + step: also counts the time this thread spends in
               prefetch_batch for each batch, i.e. what a caller of the loop
               gets (video decode / encode by this script stay excluded)
    warp_err vs input
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path
from statistics import mean, stdev

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
from dreamlite_stream.lean import (  # noqa: E402
    add_lean_args, apply_onestep_lora, apply_trt_engine, holds_clip_state, lean_worker_kwargs,
    resolve_lean_args,
)
from dreamlite_stream.metrics import compute_temporal, read_video_frames, reference_l1  # noqa: E402
from dreamlite_stream.runtime import VideoWriter, iter_video_frames  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights",
                   default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--mp4_dir", default=str(_ROOT / "assets" / "davis_mp4"))
    p.add_argument("--sequences", nargs="+", default=[
        "blackswan", "libby", "swing", "camel", "dance-twirl",
        "goat", "scooter-black", "bmx-trees", "parkour", "kite-surf",
    ])
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cond_refresh_every", type=int, default=None,
                   help="Refresh the LLLite cond every N batches. 0 = set it "
                        "once per clip and hold. Default 8, or 0 with --lean.")
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out_dir", default=str(_ROOT / "out" / "champion"))
    add_lean_args(p)
    return resolve_lean_args(p.parse_args())


@torch.no_grad()
def run_sequence(worker, args, seq_name: str, out_dir: Path):
    """Run a single sequence through an existing worker. The caller
    creates the worker once and reuses it across all sequences (prevents
    per-sequence VRAM accumulation that otherwise drives a 24-GB GPU to
    paging after ~10-15 sequences)."""
    in_path = Path(args.mp4_dir) / f"{seq_name}.mp4"
    if not in_path.exists():
        return None
    out_path = out_dir / "champion" / f"{seq_name}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / "base_oilref" / f"{seq_name}.mp4"
    base_path.parent.mkdir(parents=True, exist_ok=True)

    # Champion run with LLLite + refresh + pipelined (TE on side stream)
    timings = []
    writer = None
    fps_global = 24.0
    batch_idx = 0
    n_total = 0

    def collect_batch(it):
        nonlocal n_total, fps_global
        buf = []
        for idx, frame, fps in it:
            if idx >= args.max_frames:
                return buf, False
            fps_global = fps
            # The fast VAE path takes uint8 arrays directly; hand it those
            # rather than a PIL image it would only convert back.
            buf.append(np.asarray(frame) if args.fast_vae else frame)
            n_total += 1
            if len(buf) >= args.batch_size:
                return buf, True
        return buf, False

    iterator = iter_video_frames(str(in_path), args.size)
    # Drop partial batches: the compiled UNet + LLLite cond_emb buffer are both
    # shape-locked at batch_size, so a final batch < args.batch_size would
    # trigger recompile or dim-mismatch.
    cur_buf, more = collect_batch(iterator)
    if cur_buf and len(cur_buf) == args.batch_size:
        cur_pf = worker.prefetch_batch(cur_buf)
        while True:
            if more:
                nxt_buf, more = collect_batch(iterator)
                if len(nxt_buf) < args.batch_size:
                    nxt_buf = []  # drop partial
            else:
                nxt_buf = []
            nxt_pf = worker.prefetch_batch(nxt_buf) if nxt_buf else None
            outputs, t = worker.step_batch_with_prefetch(cur_buf, cur_pf)
            if writer is None:
                writer = VideoWriter(str(out_path), args.size, fps_global)
            for img in outputs:
                writer.write(img)
            if batch_idx >= args.warmup:
                timings.append(t)
            batch_idx += 1
            if not nxt_buf:
                break
            cur_buf, cur_pf = nxt_buf, nxt_pf
    if writer:
        writer.close()

    if not timings:
        return None
    n_meas = sum(t.n_frames for t in timings)
    sum_total = sum(t.total_ms for t in timings)
    fps_step = n_meas / (sum_total / 1000)
    # total_ms covers the step only. Each batch's prefetch call ran one
    # iteration earlier on this same thread; loop_ms adds that back.
    fps_loop = n_meas / (sum(t.loop_ms for t in timings) / 1000)

    return {
        "sequence": seq_name,
        "fps": fps_step,
        "fps_loop": fps_loop,
        "n_frames": n_total,
        "out_path": str(out_path),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)
    apply_onestep_lora(pipeline, args)

    vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
    latent_hw = args.size // vae_downsample
    block_filter = [s.strip() for s in args.lllite_blocks.split(",")] if args.lllite_blocks else None
    if args.lllite_multiplier != 0:
        controller = apply_lllite(
            pipeline.unet, cond_emb_dim=32, mlp_dim=64,
            cond_image_size=args.size, sample_size=latent_hw,
            inference_mode=True, max_batch_size=args.batch_size,
            block_filter=block_filter,
        )
        sd = load_file(args.lllite_weights)
        controller.load_state_dict(sd, strict=False)
        n_attached = len(controller.modules_dict)
        print(f"[lllite] {n_attached} hooks (blocks={args.lllite_blocks})")
        controller.to(device=args.device, dtype=torch.bfloat16)
        controller.eval()
        controller.set_multiplier(args.lllite_multiplier)
    else:
        # Multiplier 0 means no adapter: attach nothing, rather than hooks that add zero.
        controller = None
        print("[lllite] off (--lllite_multiplier 0): adapter not attached")
    apply_trt_engine(pipeline, args, controller)

    # Create ONE worker reused across all sequences to prevent per-sequence
    # VRAM accumulation that otherwise drives a 24-GB GPU to paging by
    # sequence ~10-15. Earlier versions of this script created a fresh
    # BatchedEditWorker inside run_sequence() and never freed it; the
    # accumulated prefetch buffers, side-stream state, and torch caching
    # allocator slack would saturate VRAM.
    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=args.steps, prompt=args.prompt,
    )
    worker = BatchedEditWorker(
        pipeline=pipeline, state=state, batch_size=args.batch_size,
        device=args.device, dtype=torch.bfloat16, seed=args.seed,
        compile=True, compile_mode="reduce-overhead",
        lllite_controller=controller,
        **lean_worker_kwargs(args),
    )
    print(
        f"[config] size={args.size} B={args.batch_size} K={args.steps}  "
        f"te_refresh_every={args.te_refresh_every} te_batch_one={args.te_batch_one}  "
        f"cond_refresh_every={args.cond_refresh_every}  shared_noise={args.shared_noise}  "
        f"fast_vae={args.fast_vae} vae_compile={args.vae_compile}  "
        f"onestep_lora={'yes' if args.onestep_lora else 'no'} lllite_multiplier={args.lllite_multiplier}  "
        f"trt_engine={'yes' if args.trt_engine else 'no'}"
    )

    rows = []
    print()
    for seq in args.sequences:
        print(f"=== {seq} ===")
        t_start = time.perf_counter()
        result = run_sequence(worker, args, seq, out_dir)
        # Reset per-sequence state on the reused worker (don't drag prefetch
        # buffers / LLLite cond state across sequences).
        if holds_clip_state(args):
            # A cached prompt embedding or a held cond belongs to the clip
            # it was computed from; clear all of it.
            worker.reset_sequence()
        else:
            # Default path, kept exactly as the paper runs used it: the cond
            # refresh counter deliberately carries over between sequences.
            worker.state.frame_idx = 0
            if hasattr(worker, "_last_prev_decoded"):
                worker._last_prev_decoded = None
                worker._last_prev_input_gray = None
        gc.collect()
        torch.cuda.empty_cache()
        if result is None:
            print(f"  skip {seq}: missing or empty")
            continue
        elapsed = time.perf_counter() - t_start
        # Quality vs input + (oil-paint reference will be computed later)
        in_frames = read_video_frames(str(Path(args.mp4_dir) / f"{seq}.mp4"), size=args.size)[: args.max_frames]
        out_frames = read_video_frames(result["out_path"])[: args.max_frames]
        n = min(len(in_frames), len(out_frames))
        m = compute_temporal(in_frames[:n], out_frames[:n])
        result["warp_err"] = m.warping_error
        result["con_l1"] = m.consecutive_l1
        result["consistency_ratio"] = m.consistency_ratio
        result["wall_s"] = elapsed
        rows.append(result)
        print(
            f"  fps_loop={result['fps_loop']:5.2f}  fps_step={result['fps']:5.2f}  "
            f"warp_err={m.warping_error:5.2f}  "
            f"con_l1={m.consecutive_l1:5.2f}  ratio={m.consistency_ratio:.3f}  "
            f"({elapsed:.0f}s)"
        )

    if rows:
        fpss = [r["fps"] for r in rows]
        loops = [r["fps_loop"] for r in rows]
        wes = [r["warp_err"] for r in rows]
        print()
        print("=" * 70)
        print(
            f"aggregate: fps_loop {mean(loops):.2f} ± {stdev(loops) if len(loops)>1 else 0:.2f}  "
            f"fps_step {mean(fpss):.2f} ± {stdev(fpss) if len(fpss)>1 else 0:.2f}  "
            f"warp_err {mean(wes):.2f} ± {stdev(wes) if len(wes)>1 else 0:.2f}  "
            f"N={len(rows)}"
        )

    # Save raw rows
    import json
    out_json = out_dir / "results.jsonl"
    with out_json.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
