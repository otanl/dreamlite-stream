"""Run the champion config on every OpenVE-Bench pair and write per-pair mp4s.

Mirrors ``champion_eval.py`` but iterates over ``data/openve_bench/index.jsonl``
(built by ``openve_download.py``) and sets ``state.prompt`` per pair. Outputs
go to ``out/openve/<category>/<pair_id>.mp4``; a JSONL log captures fps and
per-pair runtime for later aggregation.

This script only produces the edited videos. Judging (Seed-1.6VL / Gemini /
InternVL3.5) is the separate ``openve_judge.py`` step; aggregation into the
final LaTeX row is ``openve_table.py``.

See ``notes/openve_bench_plan.md`` for the protocol details and the honest
framing of where LLLite v3 will under-perform (Subtitles / Local Add /
Creative Edit categories are not in its training distribution).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

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
from dreamlite_stream.runtime import VideoWriter, iter_video_frames  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights",
                   default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--index",
                   default=str(_ROOT / "data" / "openve_bench" / "index.jsonl"))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cond_refresh_every", type=int, default=8)
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out_dir", default=str(_ROOT / "out" / "openve"))
    p.add_argument("--categories", nargs="*", default=None,
                   help="Filter index by category, e.g. global_style background_change. "
                        "Default: all 8.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only run the first N pairs after category filtering.")
    return p.parse_args()


def load_index(path: Path, categories: list[str] | None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if categories and rec["category"] not in categories:
                continue
            rows.append(rec)
    return rows


@torch.no_grad()
def run_pair(worker, args, rec: dict, out_root: Path):
    in_path = Path(rec["src_mp4"])
    if not in_path.exists():
        return None
    out_path = out_root / rec["category"] / f"{rec['pair_id']}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    worker.state.prompt = rec["prompt"]
    worker.state.reset()
    worker.state.frame_idx = 0
    if hasattr(worker, "_last_prev_decoded"):
        worker._last_prev_decoded = None
        worker._last_prev_input_gray = None

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
    if not (cur_buf and len(cur_buf) == args.batch_size):
        return None

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
        "pair_id": rec["pair_id"],
        "category": rec["category"],
        "fps": fps_step,
        "n_frames": n_total,
        "src_mp4": rec["src_mp4"],
        "edit_mp4": str(out_path),
    }


def main():
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = load_index(Path(args.index), args.categories)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[index] {len(rows)} pairs (categories={args.categories or 'all'})")

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
    n_attached = len(controller.modules_dict)
    print(f"[lllite] {n_attached} hooks (blocks={args.lllite_blocks})")
    controller.to(device=args.device, dtype=torch.bfloat16)
    controller.eval()
    controller.set_multiplier(1.0)

    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=args.steps,
        prompt="",
    )
    worker = BatchedEditWorker(
        pipeline=pipeline, state=state, batch_size=args.batch_size,
        device=args.device, dtype=torch.bfloat16, seed=args.seed,
        compile=True, compile_mode="reduce-overhead",
        lllite_controller=controller,
        cond_refresh_every=args.cond_refresh_every,
    )

    out_json = out_root / "results.jsonl"
    results: list[dict] = []
    with out_json.open("w", encoding="utf-8") as fout:
        for i, rec in enumerate(rows):
            t_start = time.perf_counter()
            print(f"[{i+1}/{len(rows)}] {rec['category']}/{rec['pair_id']}")
            result = run_pair(worker, args, rec, out_root)
            gc.collect()
            torch.cuda.empty_cache()
            if result is None:
                print(f"  skip: missing or empty input")
                continue
            result["wall_s"] = time.perf_counter() - t_start
            results.append(result)
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            print(f"  fps={result['fps']:5.2f}  ({result['wall_s']:.1f}s)")

    print(f"\n[saved] {out_json}  ({len(results)} pairs)")


if __name__ == "__main__":
    main()
