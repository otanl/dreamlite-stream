"""Cond-refresh interval sweep on the down_blocks champion configuration.

VRAM-safe variant: reuses one BatchedEditWorker per N value (4 total,
not 40), explicitly del-s and empties cache between sequences,
defaults to B=4 to fit comfortably in 24 GB GPUs alongside desktop
apps. Reports VRAM after every sequence for diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path
from statistics import mean, stdev

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
from dreamlite_stream.metrics import compute_temporal, read_video_frames  # noqa: E402
from dreamlite_stream.runtime import VideoWriter, iter_video_frames  # noqa: E402


def gpu_used_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3


@torch.no_grad()
def run_one_sequence(worker, state, args, seq_name, n_refresh, out_dir):
    """Run a single DAVIS sequence through an existing worker. Worker is reused
    across sequences for the same N value."""
    in_path = Path(args.mp4_dir) / f"{seq_name}.mp4"
    if not in_path.exists():
        return None
    out_path = out_dir / f"N{n_refresh}_{seq_name}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
            buf.append(frame)
            n_total += 1
            if len(buf) >= args.batch_size:
                return buf, True
        return buf, False

    iterator = iter_video_frames(str(in_path), args.size)
    cur_buf, more = collect_batch(iterator)
    if cur_buf and len(cur_buf) == args.batch_size:
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
    return {
        "sequence": seq_name, "n_refresh": n_refresh,
        "fps": fps_step, "n_frames": n_total,
        "out_path": str(out_path),
    }


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
    p.add_argument("--refresh_intervals", nargs="+", type=int, default=[1, 4, 8, 16])
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Default 4 (was 8) to fit 24 GB GPUs with desktop apps")
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out_dir", default=str(_ROOT / "out" / "cond_refresh_downblocks_sweep"))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"[lllite] {len(controller.modules_dict)} hooks (blocks={args.lllite_blocks})")

    all_rows = []
    results_path = out_dir / "results.jsonl"
    # Truncate / open append-write
    with open(results_path, "w") as _f:
        pass

    for n_refresh in args.refresh_intervals:
        print(f"\n=== N = {n_refresh} ===   [GPU {gpu_used_gb():.2f} GB]")
        # One worker reused across all sequences for this N
        state = SharedState(
            height=args.size, width=args.size,
            num_inference_steps=args.steps, prompt=args.prompt,
        )
        worker = BatchedEditWorker(
            pipeline=pipeline, state=state, batch_size=args.batch_size,
            device=args.device, dtype=torch.bfloat16, seed=args.seed,
            compile=True, compile_mode="reduce-overhead",
            lllite_controller=controller,
            cond_refresh_every=n_refresh,
        )
        for seq in args.sequences:
            t_start = time.perf_counter()
            r = run_one_sequence(worker, state, args, seq, n_refresh, out_dir / f"N{n_refresh}")
            if r is None:
                continue
            elapsed = time.perf_counter() - t_start
            in_frames = read_video_frames(str(Path(args.mp4_dir) / f"{seq}.mp4"), size=args.size)[: args.max_frames]
            out_frames = read_video_frames(r["out_path"])[: args.max_frames]
            n = min(len(in_frames), len(out_frames))
            m = compute_temporal(in_frames[:n], out_frames[:n])
            r["warp_err"] = m.warping_error
            r["con_l1"] = m.consecutive_l1
            r["consistency_ratio"] = m.consistency_ratio
            r["wall_s"] = elapsed
            r["gpu_used_gb"] = gpu_used_gb()
            all_rows.append(r)
            # Append to jsonl immediately for crash recovery
            with open(results_path, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"  {seq:18s} fps={r['fps']:5.2f}  εw={m.warping_error:5.2f}  "
                  f"GPU={r['gpu_used_gb']:.2f}GB  ({elapsed:.0f}s)")
        # explicit cleanup before next N
        del worker
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  [cleanup] GPU after del worker: {gpu_used_gb():.2f} GB")

    # Aggregate per N
    print("\n" + "=" * 70)
    print(f"{'N':>3s}  {'fps':>13s}  {'εw':>13s}  N_seq")
    by_n = {}
    for r in all_rows:
        by_n.setdefault(r["n_refresh"], []).append(r)
    for N in args.refresh_intervals:
        if N in by_n:
            rs = by_n[N]
            fpss = [r["fps"] for r in rs]
            wes = [r["warp_err"] for r in rs]
            print(f"  {N:>3d}  {mean(fpss):>5.2f} ± {stdev(fpss) if len(fpss)>1 else 0:.2f}  "
                  f"{mean(wes):>5.2f} ± {stdev(wes) if len(wes)>1 else 0:.2f}  {len(rs)}")

    print(f"\n[saved] {results_path}")


if __name__ == "__main__":
    main()
