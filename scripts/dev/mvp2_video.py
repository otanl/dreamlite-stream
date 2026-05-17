"""MVP-2: shared-state runtime with keyframe anchor + flow-warped prev latent.

Architecture:
  KeyframeWorker.generate(prompt) once -> state.keyframe_latent
  EditWorker per frame with:
      cond_image_latents = w_input * input_latent
                         + w_prev  * warp(prev_latent, flow(prev_in, in))
                         + w_kf    * keyframe_latent
      init_latents       = pure noise   (degenerate-safe; prev-init was a bad path in MVP-1)

Defaults reproduce the MVP-1.5 best speed config (compile + pipelined + 4-step)
plus a "balanced MVP-2" cond preset (input=0.4, prev=0.4, kf=0.2).

Usage:
    python scripts/mvp2_video.py \
        --input assets/davis_mp4/dance-twirl.mp4 \
        --prompt "transfer this to oil painting style, vibrant colors" \
        --keyframe_prompt "an oil painting of a person dancing, vibrant colors" \
        --w_input 0.4 --w_prev 0.4 --w_kf 0.2 \
        --steps 4 --compile --pipelined
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402

from dreamlite_stream import EditWorker, KeyframeWorker, SharedState  # noqa: E402
from dreamlite_stream.runtime import run_video, run_video_pipelined  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="out/mvp2.mp4")
    p.add_argument("--prompt", required=True, help="edit prompt applied per-frame")
    p.add_argument(
        "--keyframe_prompt", default=None,
        help="T2I prompt for the anchor keyframe; defaults to --prompt",
    )
    p.add_argument("--keyframe_save", default=None, help="optional path to save the generated keyframe PNG")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--w_input", type=float, default=0.4)
    p.add_argument("--w_prev",  type=float, default=0.4)
    p.add_argument("--w_kf",    type=float, default=0.2)
    p.add_argument("--no_keyframe", action="store_true",
                   help="skip keyframe generation (and force w_kf=0)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", default="reduce-overhead")
    p.add_argument("--pipelined", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[load] {args.model}")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)

    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=args.steps, prompt=args.prompt,
    )

    w_kf = 0.0 if args.no_keyframe else args.w_kf

    if not args.no_keyframe:
        print(f"[keyframe] generating from prompt: {(args.keyframe_prompt or args.prompt)!r}")
        kf = KeyframeWorker(
            pipeline=pipeline, state=state, device=args.device,
            dtype=torch.bfloat16, seed=args.seed,
        )
        kf_latent, kf_image = kf.generate(args.keyframe_prompt)
        if args.keyframe_save:
            Path(args.keyframe_save).parent.mkdir(parents=True, exist_ok=True)
            kf_image.save(args.keyframe_save)
            print(f"  saved {args.keyframe_save}")
        else:
            default_kf = Path(args.output).with_name("keyframe.png")
            default_kf.parent.mkdir(parents=True, exist_ok=True)
            kf_image.save(default_kf)
            print(f"  saved {default_kf}")

    worker = EditWorker(
        pipeline=pipeline, state=state, device=args.device,
        dtype=torch.bfloat16,
        init_mode="pure",  # MVP-1 found prev-init degenerates into a feedback loop
        seed=args.seed,
        compile=args.compile, compile_mode=args.compile_mode,
        w_input=args.w_input, w_prev=args.w_prev, w_kf=w_kf,
    )

    runner = run_video_pipelined if args.pipelined else run_video
    print(
        f"[run]  cond=(input={args.w_input} prev={args.w_prev} kf={w_kf})  "
        f"steps={args.steps} compile={args.compile} pipelined={args.pipelined}"
    )
    stats = runner(
        worker, in_path=args.input, out_path=args.output, size=args.size,
        max_frames=args.max_frames,
    )
    print(stats.report())


if __name__ == "__main__":
    main()
