"""Final comparison: MVP-1 baselines vs output_blend vs temporal LLLite.

Runs each method on the same input clip, dumps mp4s for quality_compare.

Order:
  1. baseline pure 4-step       (no compile, simplest reference)
  2. baseline pure 2-step       (compile_first)
  3. baseline 2-step + blend(0.85)  (= MVP-1.5 champion)
  4. temporal lllite m=0.5
  5. temporal lllite m=1.0
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import torch
from safetensors.torch import load_file

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
_LLLITE = _ROOT.parent / "dreamlite-lllite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))
sys.path.insert(0, str(_LLLITE / "src"))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402
from dreamlite_lllite import apply_lllite  # noqa: E402

from dreamlite_stream import EditWorker, SharedState  # noqa: E402
from dreamlite_stream.output_blend import OutputBlender  # noqa: E402
from dreamlite_stream.runtime import run_video_pipelined  # noqa: E402
from dreamlite_stream.workers.edit import StepTiming  # noqa: E402


@dataclass
class Config:
    name: str
    steps: int = 4
    use_blend: float = 1.0   # OutputBlender alpha; 1.0 = no blend
    use_lllite: bool = False
    lllite_multiplier: float = 1.0
    compile_first: bool = False


def avg(timings: List[StepTiming], attr: str) -> float:
    return sum(getattr(t, attr) for t in timings) / max(len(timings), 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--input", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--lllite_weights", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=30)
    p.add_argument("--cond_emb_dim", type=int, default=32)
    p.add_argument("--mlp_dim", type=int, default=64)
    p.add_argument("--out_dir", default="out/bench_final")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    # Attach LLLite once. Adapter starts with multiplier set per-config.
    vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
    latent_hw = args.size // vae_downsample
    print(f"[lllite] attaching {args.lllite_weights}")
    controller = apply_lllite(
        pipeline.unet,
        cond_emb_dim=args.cond_emb_dim,
        mlp_dim=args.mlp_dim,
        cond_image_size=args.size,
        sample_size=latent_hw,
    )
    sd = load_file(args.lllite_weights)
    controller.load_state_dict(sd, strict=True)
    controller.to(device=args.device, dtype=torch.bfloat16)
    controller.eval()

    configs = [
        Config("01_mvp1_pure_4st",          steps=4, compile_first=True),
        Config("02_mvp1_pure_2st",          steps=2),
        Config("03_mvp1_2st_blend085",      steps=2, use_blend=0.85),
        Config("04_temporal_lllite_m05_4st", steps=4, use_lllite=True, lllite_multiplier=0.5),
        Config("05_temporal_lllite_m10_4st", steps=4, use_lllite=True, lllite_multiplier=1.0),
    ]

    rows = []
    for cfg in configs:
        print(f"\n=== {cfg.name} ===")
        # Set multiplier appropriately. m=0 to fully disable when not using LLLite.
        controller.set_multiplier(cfg.lllite_multiplier if cfg.use_lllite else 0.0)

        state = SharedState(
            height=args.size, width=args.size,
            num_inference_steps=cfg.steps, prompt=args.prompt,
        )
        worker = EditWorker(
            pipeline=pipeline, state=state, device=args.device,
            dtype=torch.bfloat16, init_mode="pure", seed=args.seed,
            compile=cfg.compile_first, compile_mode="reduce-overhead",
            lllite_controller=controller if cfg.use_lllite else None,
        )
        blender = OutputBlender(alpha=cfg.use_blend) if cfg.use_blend < 1.0 else None
        out_path = str(out_dir / f"{cfg.name}.mp4")
        total = args.warmup + args.measure
        stats = run_video_pipelined(
            worker, in_path=args.input, out_path=out_path,
            size=args.size, max_frames=total, log_every=0, blender=blender,
        )
        measured = stats.timings[args.warmup:]
        if not measured:
            continue
        wall_ms = sum(t.total_ms for t in measured)
        fps = len(measured) / (wall_ms / 1000)
        rows.append((cfg, fps, measured))
        print(
            f"  -> wall_fps={fps:.2f}  total={avg(measured,'total_ms'):.0f}ms  "
            f"denoise={avg(measured,'denoise_ms'):.0f}"
        )

    print("\n" + "=" * 78)
    print(f"{'config':32s}  {'fps':>5s}  {'total':>6s}  {'denoise':>7s}")
    print("-" * 78)
    for cfg, fps, m in rows:
        print(
            f"{cfg.name:32s}  {fps:5.2f}  {avg(m,'total_ms'):6.0f}  {avg(m,'denoise_ms'):7.0f}"
        )

    print(
        f"\nNext: python scripts/quality_compare.py --input {args.input} "
        f"--bench_dir {out_dir} --size {args.size}"
    )


if __name__ == "__main__":
    main()
