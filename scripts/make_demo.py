"""Demo video through the streaming loop: input | one panel per prompt, clip after clip.

Each clip is streamed once per prompt with one reused BatchedEditWorker, batches
pipelined as in champion_eval.py (prefetch of the next batch kicked before the
current step; a partial final batch is dropped). The input panel names the clip,
the configuration and the GPU; each output panel its prompt key and the loop fps
of that run (short clips: a few batches after the warm-up, so a rougher figure
than sustained_throughput_test.py). Writes <out_dir>/<clip>.mp4 per clip and
<out_dir>/demo.mp4 with every clip in sequence; H.264 through ffmpeg when it is
on PATH, else mp4v.

    python scripts/make_demo.py --lean --size 768 --trt_engine out/trt/unet_768_b8_lora.engine \
        --out_dir out/demo_768
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch._dynamo
from PIL import Image

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
    add_lean_args, apply_onestep_lora, apply_trt_engine, lean_worker_kwargs, resolve_lean_args,
)
from dreamlite_stream.metrics import read_video_frames  # noqa: E402

# Prompt keys; all five are among those the one-step LoRAs were trained on (train_onestep_lora.py).
PROMPTS = {
    "oil": "transfer this to oil painting style, vibrant colors",
    "watercolor": "transfer this to watercolor painting style, soft edges",
    "sketch": "transfer this to pencil sketch style, fine line work",
    "anime": "transfer this to anime art style, clean cel shading",
    "ukiyoe": "transfer this to ukiyo-e woodblock print style, flat colors, line art",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights",
                   default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--mp4_dir", default=str(_ROOT / "assets" / "davis_mp4"))
    p.add_argument("--sequences", nargs="*", default=None, help="clip names (default: every mp4 in --mp4_dir)")
    p.add_argument("--prompts", nargs="+", default=["oil", "anime"], choices=sorted(PROMPTS))
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--cond_refresh_every", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--warmup", type=int, default=1, help="batches left out of the fps figure")
    p.add_argument("--out_dir", required=True)
    add_lean_args(p)
    return resolve_lean_args(p.parse_args())


def label(img: np.ndarray, text: str, bottom: bool = False) -> np.ndarray:
    """img with `text` in white on a black box, top-left (or bottom-left)."""
    img = img.copy()
    scale = img.shape[0] / 1200
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y0 = img.shape[0] - (h + base + 14) if bottom else 0
    cv2.rectangle(img, (0, y0), (w + 16, y0 + h + base + 14), (0, 0, 0), -1)
    cv2.putText(img, text, (8, y0 + h + 7), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save(path: Path, frames, fps: float) -> None:
    h, w = frames[0].shape[:2]
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        return
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed for {path}")


@torch.no_grad()
def stream(worker, frames, batch_size: int, warmup: int):
    """Outputs for `frames` (a multiple of batch_size) and the loop fps after `warmup` batches."""
    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    outs, timings = [], []
    pf = worker.prefetch_batch(batches[0])
    for i, cur in enumerate(batches):
        nxt = worker.prefetch_batch(batches[i + 1]) if i + 1 < len(batches) else None
        res, t = worker.step_batch_with_prefetch(cur, pf)
        outs += [np.asarray(o) for o in res]
        if i >= warmup:
            timings.append(t)
        pf = nxt
    timings = timings or [t]
    return outs, sum(t.n_frames for t in timings) / (sum(t.loop_ms for t in timings) / 1000)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    apply_onestep_lora(pipeline, args)
    controller = None
    if args.lllite_multiplier != 0:
        latent_hw = args.size // 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
        controller = apply_lllite(
            pipeline.unet, cond_emb_dim=32, mlp_dim=64, cond_image_size=args.size, sample_size=latent_hw,
            inference_mode=True, max_batch_size=args.batch_size,
            block_filter=[s.strip() for s in args.lllite_blocks.split(",")] if args.lllite_blocks else None,
        )
        controller.load_state_dict(load_file(args.lllite_weights), strict=False)
        controller.to(device=args.device, dtype=torch.bfloat16).eval()
        controller.set_multiplier(args.lllite_multiplier)
    apply_trt_engine(pipeline, args, controller)

    state = SharedState(height=args.size, width=args.size, num_inference_steps=args.steps,
                        prompt=PROMPTS[args.prompts[0]])
    worker = BatchedEditWorker(
        pipeline=pipeline, state=state, batch_size=args.batch_size, device=args.device, dtype=torch.bfloat16,
        seed=args.seed, compile=True, compile_mode="reduce-overhead", lllite_controller=controller,
        **lean_worker_kwargs(args),
    )

    lora = " + LoRA" if args.onestep_lora or args.trt_engine else ""
    backend = "TensorRT" if args.trt_engine else "PyTorch"
    gpu = torch.cuda.get_device_name().replace("NVIDIA ", "").replace("GeForce ", "")
    footer = f"DreamLite-mobile {args.steps} step{'s' if args.steps > 1 else ''}{lora}, {args.size}px, {backend}, {gpu}"
    clips = args.sequences or sorted(f.stem for f in Path(args.mp4_dir).glob("*.mp4"))

    def load(clip):
        frames = read_video_frames(str(Path(args.mp4_dir) / f"{clip}.mp4"), size=args.size)[:args.max_frames]
        return frames[:len(frames) // args.batch_size * args.batch_size]

    def inputs(frames):  # the fast VAE path takes uint8 arrays, the default one PIL images
        return frames if args.fast_vae else [Image.fromarray(f) for f in frames]

    # Compile and capture once before anything is timed.
    stream(worker, inputs(load(clips[0])[:2 * args.batch_size]), args.batch_size, 0)
    worker.reset_sequence()
    demo, fps = [], 24.0
    for clip in clips:
        frames = load(clip)
        fps = cv2.VideoCapture(str(Path(args.mp4_dir) / f"{clip}.mp4")).get(cv2.CAP_PROP_FPS) or 24.0
        panels = [[label(label(f, f"input: {clip}"), footer, bottom=True) for f in frames]]
        for key in args.prompts:
            worker.state.prompt = PROMPTS[key]
            worker.reset_sequence()
            outs, loop_fps = stream(worker, inputs(frames), args.batch_size, args.warmup)
            panels.append([label(o, f"{key}: {loop_fps:.0f} fps loop") for o in outs])
            print(f"[demo] {clip:<14} {key:<6} {len(frames)} frames, loop {loop_fps:.1f} fps", flush=True)
        rows = [np.concatenate(fs, axis=1) for fs in zip(*panels)]
        save(out_dir / f"{clip}.mp4", rows, fps)
        demo += rows
    save(out_dir / "demo.mp4", demo, fps)
    print(f"[demo] {out_dir / 'demo.mp4'}: {len(clips)} clips, {len(demo)} frames", flush=True)


if __name__ == "__main__":
    main()
