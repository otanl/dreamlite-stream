"""Run all key configs on all 10 DAVIS sequences for paper Table 1+2.

Configs (in order; some share state to amortize compile cost):
  baseline_eager      : MVP-0 reference (4-step, eager, no LLLite, no batching)
  compile_2st_blend   : MVP-1.5 (compile + 2-step + post-hoc blend)
  base_b8_1st_pipe    : speed pipeline base (no LLLite)
  base_b16_1st_pipe   : same at B=16
  lllite_v3_eager     : LLLite + eager (no compile)
  lllite_v3_compile   : compile-friendly LLLite + B=8
  champion_b8         : compile + LLLite + refresh8 + blocks_down + B=8
  champion_b16        : same at B=16  (paper quality champion)
  champion_b16_nf4    : champion + 4-bit TE (deployment champion)

Per config × per sequence, we record FPS, warp_err, consecutive_l1, ratio.
For LLLite-using configs we also record reference_l1 vs an LLLite-only ref.

Output:
  out/comprehensive_ablation/results.jsonl   (raw rows)
  out/comprehensive_ablation/summary.csv     (aggregated mean±std)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import List, Optional

import torch
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
_LLLITE = _ROOT.parent / "dreamlite-lllite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))
sys.path.insert(0, str(_LLLITE / "src"))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402

from dreamlite_stream import BatchedEditWorker, EditWorker, SharedState  # noqa: E402
from dreamlite_stream.metrics import compute_temporal, read_video_frames, reference_l1  # noqa: E402
from dreamlite_stream.output_blend import OutputBlender  # noqa: E402
from dreamlite_stream.runtime import (  # noqa: E402
    VideoWriter, iter_video_frames, run_video, run_video_pipelined,
)


@dataclass
class Cfg:
    name: str
    mode: str   # 'eager' | 'edit_compile' | 'batched' | 'batched_lllite'
    compile: bool = False
    pipelined: bool = True
    steps: int = 1
    batch_size: int = 8
    lllite: bool = False
    cond_refresh: int = 1
    lllite_blocks: Optional[str] = None  # e.g. 'down_blocks'
    blend_alpha: float = 1.0
    te_quant: Optional[str] = None
    use_lllite_quality_ref: bool = False  # report L1 vs lllite-only reference


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
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out_dir", default=str(_ROOT / "out" / "comprehensive_ablation"))
    p.add_argument("--configs", nargs="+", default=None,
                   help="restrict to these config names (default: all)")
    return p.parse_args()


CONFIGS: List[Cfg] = [
    # Baselines (no LLLite)
    Cfg("baseline_eager",       mode="eager",       compile=False, pipelined=False, steps=4, batch_size=1),
    Cfg("compile_2st_blend",    mode="edit_compile", compile=True, pipelined=True,  steps=2, batch_size=1, blend_alpha=0.85),
    # Speed pipeline (no LLLite)
    Cfg("base_b8_1st_pipe",     mode="batched",     compile=True,  pipelined=True,  steps=1, batch_size=8),
    Cfg("base_b16_1st_pipe",    mode="batched",     compile=True,  pipelined=True,  steps=1, batch_size=16),
    # LLLite path
    Cfg("lllite_v3_eager",      mode="edit_compile", compile=False, pipelined=False, steps=1, batch_size=1, lllite=True, use_lllite_quality_ref=True),
    Cfg("lllite_v3_compile",    mode="batched_lllite", compile=True, pipelined=True, steps=1, batch_size=8, lllite=True, use_lllite_quality_ref=True),
    Cfg("champion_b8",          mode="batched_lllite", compile=True, pipelined=True, steps=1, batch_size=8,  lllite=True, cond_refresh=8, lllite_blocks="down_blocks"),
    Cfg("champion_b16",         mode="batched_lllite", compile=True, pipelined=True, steps=1, batch_size=16, lllite=True, cond_refresh=8, lllite_blocks="down_blocks"),
    Cfg("champion_b16_nf4",     mode="batched_lllite", compile=True, pipelined=True, steps=1, batch_size=16, lllite=True, cond_refresh=8, lllite_blocks="down_blocks", te_quant="nf4"),
]


def load_pipeline(args, te_quant: Optional[str]):
    dtype = torch.bfloat16
    pipeline = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    if te_quant:
        from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        if te_quant == "int8":
            qc = BitsAndBytesConfig(load_in_8bit=True)
        else:
            qc = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=te_quant, bnb_4bit_use_double_quant=True,
            )
        te_path = Path(args.model) / "text_encoder"
        te_q = Qwen3VLForConditionalGeneration.from_pretrained(
            te_path, quantization_config=qc, device_map={"": args.device},
            torch_dtype=torch.float16,
        )
        del pipeline.text_encoder
        torch.cuda.empty_cache()
        pipeline.text_encoder = te_q
    return pipeline, dtype


def _unwrap_compiled(unet):
    """Unwrap nested torch.compile wrappers to the underlying nn.Module."""
    while hasattr(unet, "_orig_mod"):
        unet = unet._orig_mod
    return unet


def attach_lllite(pipeline, args, max_batch: int, blocks: Optional[str], inference_mode: bool):
    from dreamlite_lllite import apply_lllite
    from safetensors.torch import load_file
    vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
    latent_hw = args.size // vae_downsample
    block_filter = [s.strip() for s in blocks.split(",")] if blocks else None
    target_unet = _unwrap_compiled(pipeline.unet)
    controller = apply_lllite(
        target_unet, cond_emb_dim=32, mlp_dim=64,
        cond_image_size=args.size, sample_size=latent_hw,
        inference_mode=inference_mode, max_batch_size=max_batch,
        block_filter=block_filter,
    )
    sd = load_file(args.lllite_weights)
    controller.load_state_dict(sd, strict=False)
    controller.to(device=args.device, dtype=torch.bfloat16)
    controller.eval()
    controller.set_multiplier(1.0)
    return controller


def run_one(pipeline, controller, cfg: Cfg, seq: str, args, out_dir: Path):
    in_path = Path(args.mp4_dir) / f"{seq}.mp4"
    if not in_path.exists():
        return None
    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=cfg.steps, prompt=args.prompt,
    )
    out_path = out_dir / cfg.name / f"{seq}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timings = []
    t_start = time.perf_counter()

    if cfg.mode in ("eager", "edit_compile"):
        # Single-frame EditWorker path
        worker = EditWorker(
            pipeline=pipeline, state=state, device=args.device,
            dtype=pipeline.unet.dtype if hasattr(pipeline.unet, "dtype") else torch.bfloat16,
            init_mode="pure", seed=args.seed,
            compile=cfg.compile, lllite_controller=controller if cfg.lllite else None,
        )
        runner = run_video_pipelined if cfg.pipelined else run_video
        blender = OutputBlender(alpha=cfg.blend_alpha) if cfg.blend_alpha < 1.0 else None
        stats = runner(
            worker, in_path=str(in_path), out_path=str(out_path),
            size=args.size, max_frames=args.warmup + args.max_frames, log_every=0,
            blender=blender,
        )
        measured = stats.timings[args.warmup:]
        if not measured:
            return None
        wall_ms = sum(t.total_ms for t in measured)
        fps = len(measured) / (wall_ms / 1000)
    else:  # 'batched' or 'batched_lllite'
        worker = BatchedEditWorker(
            pipeline=pipeline, state=state, batch_size=cfg.batch_size,
            device=args.device, dtype=torch.bfloat16, seed=args.seed,
            compile=cfg.compile, compile_mode="reduce-overhead",
            lllite_controller=controller if cfg.lllite else None,
            cond_refresh_every=cfg.cond_refresh,
        )
        writer = None
        fps_global = 24.0
        batch_idx = 0

        def collect_batch(it, max_total):
            buf = []
            for idx, frame, fps in it:
                if idx >= max_total: return buf, False
                nonlocal fps_global
                fps_global = fps
                buf.append(frame)
                if len(buf) >= cfg.batch_size:
                    return buf, True
            return buf, False

        max_total = args.warmup + args.max_frames
        iterator = iter_video_frames(str(in_path), args.size)
        if cfg.pipelined:
            cur_buf, more = collect_batch(iterator, max_total)
            if cur_buf and len(cur_buf) == cfg.batch_size:
                cur_pf = worker.prefetch_batch(cur_buf)
                while True:
                    if more:
                        nxt_buf, more = collect_batch(iterator, max_total)
                        if len(nxt_buf) < cfg.batch_size:
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
        else:
            while True:
                buf, more = collect_batch(iterator, max_total)
                if not buf or len(buf) < cfg.batch_size:
                    break
                outputs, t = worker.step_batch(buf)
                if writer is None:
                    writer = VideoWriter(str(out_path), args.size, fps_global)
                for img in outputs:
                    writer.write_pil(img)
                if batch_idx >= args.warmup:
                    timings.append(t)
                batch_idx += 1
                if not more:
                    break
        if writer:
            writer.close()
        if not timings:
            return None
        n_meas = sum(t.n_frames for t in timings)
        sum_total = sum(t.total_ms for t in timings)
        fps = n_meas / (sum_total / 1000)

    # Quality
    in_frames = read_video_frames(str(in_path), size=args.size)[: args.warmup + args.max_frames]
    out_frames = read_video_frames(str(out_path))[: args.warmup + args.max_frames]
    n = min(len(in_frames), len(out_frames))
    m = compute_temporal(in_frames[:n], out_frames[:n]) if n >= 2 else None

    elapsed = time.perf_counter() - t_start
    return {
        "config": cfg.name,
        "sequence": seq,
        "fps": fps,
        "warp_err": (m.warping_error if m else 0.0),
        "consecutive_l1": (m.consecutive_l1 if m else 0.0),
        "consistency_ratio": (m.consistency_ratio if m else 0.0),
        "wall_s": elapsed,
        "out_path": str(out_path),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    # Append-only: skip (cfg, seq) combos already done so we can resume on crash.
    done_keys = set()
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done_keys.add((r["config"], r["sequence"]))
        print(f"[resume] already have {len(done_keys)} (cfg, seq) results")

    configs_to_run = [c for c in CONFIGS if (args.configs is None or c.name in args.configs)]
    print(f"Running {len(configs_to_run)} configs × {len(args.sequences)} sequences")

    # Group configs by te_quant so we don't reload pipeline unnecessarily.
    # Within a quant group, configs may need different LLLite max_batch -> reload controller.
    from itertools import groupby
    sorted_cfgs = sorted(configs_to_run, key=lambda c: (str(c.te_quant), c.batch_size))

    pipeline = None
    pipeline_quant = "INVALID"
    controller = None
    controller_key = None  # (max_batch, blocks)
    raw_unet = None  # original nn.Module reference for resetting between configs

    for cfg in sorted_cfgs:
        # Reload pipeline if quant changed
        cur_quant = str(cfg.te_quant)
        if cur_quant != pipeline_quant:
            print(f"\n[load pipeline] te_quant={cfg.te_quant}")
            del pipeline
            pipeline = None
            controller = None
            controller_key = None
            raw_unet = None
            torch.cuda.empty_cache()
            pipeline, _ = load_pipeline(args, cfg.te_quant)
            pipeline_quant = cur_quant
            raw_unet = pipeline.unet  # save original

        # Reset pipeline.unet to raw between configs to avoid chained compiles
        pipeline.unet = raw_unet

        # Manage LLLite controller (re-attach if max_batch / blocks change)
        max_batch = max(cfg.batch_size, 1)
        new_key = (max_batch, cfg.lllite_blocks) if cfg.lllite else None
        if new_key != controller_key:
            if controller is not None:
                # Detach existing controller from UNet
                from dreamlite_lllite import remove_lllite
                try:
                    remove_lllite(controller)
                except Exception:
                    pass
                del controller
                controller = None
                torch.cuda.empty_cache()
            if cfg.lllite:
                print(f"[lllite] attach max_batch={max_batch} blocks={cfg.lllite_blocks}")
                controller = attach_lllite(
                    pipeline, args, max_batch=max_batch,
                    blocks=cfg.lllite_blocks, inference_mode=True,
                )
            controller_key = new_key

        for seq in args.sequences:
            if (cfg.name, seq) in done_keys:
                continue
            print(f"  {cfg.name:24s}  {seq:14s} ", end="", flush=True)
            try:
                r = run_one(pipeline, controller, cfg, seq, args, out_dir)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            if r is None:
                print("  (skipped)")
                continue
            print(f"fps={r['fps']:5.2f}  warp_err={r['warp_err']:5.2f}")
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")

    # Aggregate
    rows = []
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)

    print("\n" + "=" * 80)
    print(f"{'config':24s}  {'fps avg±std':>14s}  {'warp_err avg±std':>20s}  N")
    print("-" * 80)
    summary_rows = []
    for cfg in configs_to_run:
        rs = by_cfg.get(cfg.name, [])
        if not rs:
            continue
        fps_vals = [r["fps"] for r in rs]
        we_vals = [r["warp_err"] for r in rs]
        f_avg, f_std = mean(fps_vals), stdev(fps_vals) if len(fps_vals) > 1 else 0.0
        w_avg, w_std = mean(we_vals), stdev(we_vals) if len(we_vals) > 1 else 0.0
        print(f"{cfg.name:24s}  {f_avg:6.2f} ± {f_std:5.2f}  "
              f"{w_avg:9.2f} ± {w_std:7.2f}  {len(rs)}")
        summary_rows.append({
            "config": cfg.name, "n": len(rs),
            "fps_avg": f_avg, "fps_std": f_std,
            "warp_err_avg": w_avg, "warp_err_std": w_std,
        })

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", "n", "fps_avg", "fps_std", "warp_err_avg", "warp_err_std"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[saved] {results_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
