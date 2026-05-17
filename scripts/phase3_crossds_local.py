"""Phase 3: cross-dataset evaluation.

Downloads Big Buck Bunny (Blender, CC-BY), Sintel (Blender, CC-BY), and
Jellyfish (test-videos.co.uk natural footage) and runs champion_eval on
3-second segments. Cross-dataset because the content spans 3D animation
+ natural underwater footage, both very different from DAVIS-2017
natural single-shot videos.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
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

# Each entry: (output_name, source_url, segment_start_seconds, segment_duration)
# Cross-dataset set spanning 7 distinct provenances:
#   Blender movies (BBB×3 + Sintel×1), test-videos.co.uk Jellyfish×3,
#   MDN CC0 (flower, friday), learning-container, Intel IoT samples ×5.
SOURCES = [
    # --- Animated 3D (Blender, CC-BY) — reduced from BBB×6 to BBB×3 ---
    ("bbb_field",     "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4", 120, 3),
    ("bbb_squirrel",  "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4", 240, 3),
    ("bbb_action",    "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4", 480, 3),
    ("sintel_dragon", "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4", 30, 3),
    # --- Natural footage: underwater (test-videos.co.uk) ---
    ("jellyfish_a",   "https://test-videos.co.uk/vids/jellyfish/mp4/h264/720/Jellyfish_720_10s_5MB.mp4", 0, 3),
    ("jellyfish_b",   "https://test-videos.co.uk/vids/jellyfish/mp4/h264/720/Jellyfish_720_10s_5MB.mp4", 4, 3),
    ("jellyfish_c",   "https://test-videos.co.uk/vids/jellyfish/mp4/h264/720/Jellyfish_720_10s_5MB.mp4", 7, 3),
    # --- Natural footage: CC0 single-clip (MDN, LC) ---
    ("mdn_flower",    "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4", 0, 3),
    ("mdn_friday",    "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/friday.mp4", 0, 3),
    ("lc_sample",     "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4", 5, 3),
    # --- Natural footage: Intel IoT sample suite (surveillance / retail / indoor) ---
    ("intel_people",  "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4", 2, 3),
    ("intel_face",    "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4", 2, 3),
    ("intel_store",   "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/store-aisle-detection.mp4", 5, 3),
    ("intel_headpose","https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/head-pose-face-detection-female-and-male.mp4", 5, 3),
    ("intel_bottle",  "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/bottle-detection.mp4", 0, 3),
]


def gpu_used_gb():
    return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


def download(url: str, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 1000:
        return True
    print(f"  download {out_path.name} <- {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        out_path.write_bytes(data)
        return True
    except Exception as e:
        print(f"    WARN failed: {type(e).__name__}: {e}")
        return False


def extract_segment(src: Path, out: Path, start_s: float, dur_s: float,
                    size: int, fps: int):
    if out.exists():
        return True
    cmd = [
        "ffmpeg", "-ss", str(start_s), "-i", str(src),
        "-t", str(dur_s),
        "-vf", f"crop='min(iw,ih)':'min(iw,ih)',scale={size}:{size}",
        "-r", str(fps), "-an", "-y",
        str(out),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ffmpeg WARN {out.name}: {r.stderr[-200:]}")
            return False
        return True
    except FileNotFoundError:
        print("  ERROR ffmpeg not found in PATH")
        return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights",
                   default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--raw_dir", default=str(_ROOT / "assets" / "crossds_raw"))
    p.add_argument("--mp4_dir", default=str(_ROOT / "assets" / "crossds_512"))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--cond_refresh_every", type=int, default=8)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out_dir", default=str(_ROOT / "out" / "champion_crossds"))
    return p.parse_args()


def main():
    args = parse_args()
    print("[phase 3] cross-dataset eval (Blender + test-videos.co.uk)")
    raw_dir = Path(args.raw_dir)
    mp4_dir = Path(args.mp4_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    mp4_dir.mkdir(parents=True, exist_ok=True)

    # Download source movies (dedup by URL)
    print(f"\n[download] -> {raw_dir}")
    url_to_local = {}
    for key, url, _, _ in SOURCES:
        if url in url_to_local:
            continue
        fname = url.split("/")[-1]
        local = raw_dir / fname
        if download(url, local):
            url_to_local[url] = local

    print(f"\n[extract segments] -> {mp4_dir}")
    for key, url, start_s, dur_s in SOURCES:
        if url not in url_to_local:
            continue
        src = url_to_local[url]
        out = mp4_dir / f"{key}.mp4"
        if extract_segment(src, out, start_s, dur_s, args.size, args.fps):
            print(f"  extracted {key} ({start_s}s + {dur_s}s)")

    sequences = sorted([p.stem for p in mp4_dir.glob("*.mp4")])
    print(f"\n[sequences] {sequences}")
    if not sequences:
        print("[abort] no sequences ready")
        return

    print(f"\n[load] {args.model}")
    from dreamlite import DreamLiteMobilePipeline
    from dreamlite_lllite import apply_lllite
    from safetensors.torch import load_file
    from dreamlite_stream import BatchedEditWorker, SharedState
    from dreamlite_stream.metrics import compute_temporal, read_video_frames
    from dreamlite_stream.runtime import VideoWriter, iter_video_frames

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
    print(f"[lllite] {len(controller.modules_dict)} hooks")

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    with open(results_path, "w") as _f:
        pass
    rows = []

    for seq in sequences:
        in_path = mp4_dir / f"{seq}.mp4"
        out_path = out_dir / "champion" / f"{seq}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        timings = []
        writer = None
        fps_global = args.fps
        batch_idx = 0
        n_total = 0
        t_start = time.perf_counter()

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
            print(f"  {seq}: no timings (clip too short for warmup)")
            continue
        elapsed = time.perf_counter() - t_start

        in_frames = read_video_frames(str(in_path), size=args.size)[: args.max_frames]
        out_frames = read_video_frames(str(out_path))[: args.max_frames]
        n = min(len(in_frames), len(out_frames))
        m = compute_temporal(in_frames[:n], out_frames[:n])
        n_meas = sum(t.n_frames for t in timings)
        sum_total = sum(t.total_ms for t in timings)
        fps_step = n_meas / (sum_total / 1000)

        row = {
            "sequence": seq, "fps": fps_step, "n_frames": n_total,
            "out_path": str(out_path),
            "warp_err": m.warping_error, "con_l1": m.consecutive_l1,
            "consistency_ratio": m.consistency_ratio,
            "wall_s": elapsed, "gpu_used_gb": gpu_used_gb(),
        }
        rows.append(row)
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {seq:18s} fps={fps_step:5.2f}  εw={m.warping_error:5.2f}  "
              f"ratio={m.consistency_ratio:.3f}  GPU={row['gpu_used_gb']:.2f}GB  ({elapsed:.0f}s)")

    if rows:
        fpss = [r["fps"] for r in rows]
        wes = [r["warp_err"] for r in rows]
        cons = [r["consistency_ratio"] for r in rows]
        print()
        print("=" * 70)
        print(f"Cross-dataset aggregate (Blender + test-videos.co.uk):")
        print(f"  fps {mean(fpss):.2f} ± {stdev(fpss) if len(fpss)>1 else 0:.2f}")
        print(f"  warp_err {mean(wes):.2f} ± {stdev(wes) if len(wes)>1 else 0:.2f}")
        print(f"  consistency_ratio {mean(cons):.3f}")
        print(f"  N={len(rows)}")
        print(f"[saved] {results_path}")


if __name__ == "__main__":
    main()
