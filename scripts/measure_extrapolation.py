"""Measure how close pure flow-extrapolation of the previous output's latent
is to the actual denoised next-frame output.

This is the upper-bound viability test for the speculative-frame path:

    latent_speculated = warp_latent(latent_t, flow_in_t->t+1)
    latent_actual     = denoise(input_t+1, prompt)

If `dist(speculated, actual)` is small for most pairs, then on hit frames we
can skip the full denoise and use the extrapolated latent directly (decode
only). The expected runtime improvement is roughly:

    fps_spec = 1 / [(1-h) * t_full + h * t_cheap]

where h is the hit rate and t_cheap = (flow + warp + decode) ~= 20-50ms.

We report:
  - per-frame latent L2 and cosine distances
  - per-frame DECODED-image L1 and warping error vs actual
  - hit rates at thresholds {25%, 50%, 75%} percentiles of the distribution
  - dump a side-by-side grid for visual inspection at various flow magnitudes
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
_LLLITE = _ROOT.parent / "dreamlite-lllite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))
sys.path.insert(0, str(_LLLITE / "src"))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402

from dreamlite_stream import EditWorker, SharedState  # noqa: E402
from dreamlite_stream import flow as flowlib  # noqa: E402
from dreamlite_stream import pipeline_ops as ops  # noqa: E402
from dreamlite_stream.metrics import read_video_frames, make_grid  # noqa: E402


@dataclass
class PairStat:
    seq: str
    frame_idx: int
    flow_mag_mean: float
    flow_mag_max: float
    latent_l2: float
    latent_cos: float
    decoded_l1: float


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--inputs", nargs="+", required=True,
                   help="DAVIS-style mp4 files (e.g. dance-twirl.mp4)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames_per_seq", type=int, default=40)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--lllite_weights", default=None,
                   help="optional trained temporal LLLite to enable on the base — "
                        "tests whether a consistent base makes speculation viable")
    p.add_argument("--lllite_multiplier", type=float, default=1.0)
    p.add_argument("--out_dir", default="out/extrapolation")
    return p.parse_args()


@torch.no_grad()
def stylize_one(worker: EditWorker, frame_rgb: np.ndarray):
    """Run one frame through worker.step, returning (latent, decoded_RGB)."""
    from PIL import Image
    out_pil, _ = worker.step(Image.fromarray(frame_rgb))
    decoded = np.asarray(out_pil.convert("RGB"))
    latent = worker.state.last_latent()  # last latent stored by step()
    return latent, decoded


@torch.no_grad()
def decode_one(pipeline, latent: torch.Tensor) -> np.ndarray:
    images = ops.decode_latent(pipeline, latent, output_type="pil")
    return np.asarray(images[0].convert("RGB"))


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm((a - b).flatten()).cpu())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    return float(torch.dot(af, bf) / (af.norm() * bf.norm() + 1e-8))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)
    controller = None
    if args.lllite_weights is not None:
        from dreamlite_lllite import apply_lllite
        from safetensors.torch import load_file
        vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
        latent_hw = args.size // vae_downsample
        controller = apply_lllite(
            pipeline.unet, cond_emb_dim=32, mlp_dim=64,
            cond_image_size=args.size, sample_size=latent_hw,
        )
        controller.load_state_dict(load_file(args.lllite_weights), strict=True)
        controller.to(device=args.device, dtype=torch.bfloat16)
        controller.eval()
        controller.set_multiplier(args.lllite_multiplier)
        print(f"[lllite] enabled multiplier={args.lllite_multiplier}")
    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=args.steps, prompt=args.prompt,
    )
    worker = EditWorker(
        pipeline=pipeline, state=state, device=args.device,
        dtype=torch.bfloat16, init_mode="pure", seed=args.seed,
        compile=not args.no_compile, compile_mode="reduce-overhead",
        lllite_controller=controller,
    )

    all_stats: List[PairStat] = []
    grid_panels: List[Tuple[str, np.ndarray]] = []

    for in_path in args.inputs:
        seq_name = Path(in_path).stem
        frames = read_video_frames(in_path, size=args.size)[: args.max_frames_per_seq]
        if len(frames) < 2:
            continue
        print(f"\n=== {seq_name}  frames={len(frames)} ===")

        # Stylize the first frame to seed the loop.
        state.reset()
        prev_lat, prev_decoded = stylize_one(worker, frames[0])
        prev_input_gray = flowlib.to_gray(frames[0])

        for t in range(1, len(frames)):
            curr = frames[t]
            curr_gray = flowlib.to_gray(curr)

            # 1. Compute flow on inputs and warp prev_lat → speculated latent
            flow = flowlib.farneback_flow(prev_input_gray, curr_gray)
            flow_mag = np.linalg.norm(flow, axis=2)
            speculated_lat = flowlib.warp_latent(
                prev_lat, flow, pipeline.vae_scale_factor,
            )

            # 2. Run actual denoise to get ground truth.
            actual_lat, actual_decoded = stylize_one(worker, curr)

            # 3. Distances in latent space
            l2 = _l2(speculated_lat, actual_lat)
            cs = _cos(speculated_lat, actual_lat)

            # 4. Decode the speculation for image-space comparison
            speculated_decoded = decode_one(pipeline, speculated_lat)
            decoded_l1 = float(np.mean(np.abs(
                speculated_decoded.astype(np.float32) - actual_decoded.astype(np.float32),
            )))

            stat = PairStat(
                seq=seq_name, frame_idx=t,
                flow_mag_mean=float(flow_mag.mean()),
                flow_mag_max=float(flow_mag.max()),
                latent_l2=l2, latent_cos=cs, decoded_l1=decoded_l1,
            )
            all_stats.append(stat)

            # 5. Optionally collect a grid panel (every 5th pair, capped)
            if t % 5 == 0 and len(grid_panels) < 18:
                cap = (
                    f"{seq_name} t={t} "
                    f"flow={stat.flow_mag_mean:.1f}/{stat.flow_mag_max:.0f} "
                    f"L1={stat.decoded_l1:.1f}"
                )
                triple = np.concatenate([
                    cv2.cvtColor(curr, cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(actual_decoded, cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(speculated_decoded, cv2.COLOR_RGB2BGR),
                ], axis=1)
                grid_panels.append((cap, triple))

            prev_lat = actual_lat
            prev_decoded = actual_decoded
            prev_input_gray = curr_gray

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    if not all_stats:
        print("(no stats)")
        return

    flow_means   = np.array([s.flow_mag_mean for s in all_stats])
    flow_maxes   = np.array([s.flow_mag_max  for s in all_stats])
    l2s          = np.array([s.latent_l2     for s in all_stats])
    coss         = np.array([s.latent_cos    for s in all_stats])
    decoded_l1s  = np.array([s.decoded_l1    for s in all_stats])

    print("\n" + "=" * 70)
    print(f"pairs measured: {len(all_stats)}")
    print(f"flow_mag_mean:   median={np.median(flow_means):.2f}  p90={np.percentile(flow_means, 90):.2f}")
    print(f"flow_mag_max:    median={np.median(flow_maxes):.2f}  p90={np.percentile(flow_maxes, 90):.2f}")
    print(f"latent_L2:       median={np.median(l2s):.2f}  p25={np.percentile(l2s, 25):.2f}  p75={np.percentile(l2s, 75):.2f}")
    print(f"latent_cos:      median={np.median(coss):.4f} p25={np.percentile(coss, 25):.4f} p75={np.percentile(coss, 75):.4f}")
    print(f"decoded_L1:      median={np.median(decoded_l1s):.2f}  p25={np.percentile(decoded_l1s, 25):.2f}  p75={np.percentile(decoded_l1s, 75):.2f}")

    # Hit-rate analysis: at each decoded_L1 threshold, what fraction of pairs
    # would be "good enough" (i.e. speculation acceptable)?
    print("\nHit-rate vs decoded_L1 threshold:")
    for thr in [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]:
        h = float(np.mean(decoded_l1s < thr))
        print(f"  L1 < {thr:5.1f}:  hit_rate = {h*100:5.1f}%")

    # Save numeric data
    with (out_dir / "stats.jsonl").open("w", encoding="utf-8") as f:
        for s in all_stats:
            f.write(json.dumps(s.__dict__) + "\n")
    print(f"\n[saved] {out_dir/'stats.jsonl'}")

    # Save grid (rows are sorted by flow_mag for readability)
    grid_panels.sort(key=lambda p: float(p[0].split("flow=")[1].split("/")[0]))
    if grid_panels:
        h, w, _ = grid_panels[0][1].shape
        rows = []
        for cap, panel in grid_panels:
            cap_bar = np.full((28, w, 3), 32, dtype=np.uint8)
            cv2.putText(cap_bar, cap, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (240, 240, 240), 1, cv2.LINE_AA)
            rows.append(np.concatenate([cap_bar, panel], axis=0))
        grid = np.concatenate(rows, axis=0)
        (out_dir / "grid.png").parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "grid.png"), grid)
        print(f"[saved] {out_dir/'grid.png'}  layout: input | actual | speculated  (rows sorted by flow magnitude)")


if __name__ == "__main__":
    main()
