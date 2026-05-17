"""MVP-3: batched-frame inference for streaming.

Buffers `--batch_size` frames before each UNet call, exploiting:
  - Qwen3-VL TE sub-linear batch scaling (~3-4x at B=4): the heavy edit-mode
    encoder is the per-frame bottleneck; batching hides ~75% of it.
  - VAE encode/decode sub-linear scaling (similar reason: uncompiled).
  - Compiled UNet (already saturated): batch=4 only ~1.2x but free since the
    other stages dominate.

Trade-off: +(N-1) frame buffering latency.

Usage:
    python scripts/mvp3_batched.py \
        --input  assets/davis_mp4/dance-twirl.mp4 \
        --output out/mvp3_b4.mp4 \
        --prompt "transfer this to oil painting style, vibrant colors" \
        --size 512 --steps 2 --batch_size 4 --compile
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import List

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

from dreamlite_stream import BatchedEditWorker, SharedState  # noqa: E402
from dreamlite_stream.runtime import VideoWriter, iter_video_frames  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="out/mvp3.mp4")
    p.add_argument("--prompt", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", default="reduce-overhead")
    p.add_argument("--pipelined", action="store_true",
                   help="overlap TE+VAE_enc of NEXT batch with denoise of CURRENT batch (CUDA streams)")
    p.add_argument("--warmup", type=int, default=1,
                   help="warmup BATCHES to discard (compile cost on first batch is huge)")
    p.add_argument("--lllite_weights", default=None,
                   help="optional trained temporal LLLite (.safetensors) — adds adapter delta per frame")
    p.add_argument("--lllite_multiplier", type=float, default=1.0)
    p.add_argument("--fixed_noise", action="store_true",
                   help="use the same noise pattern every frame (streamdiffusion-mac trick; reduces flicker for free)")
    p.add_argument("--cond_refresh_every", type=int, default=1,
                   help="rebuild LLLite cond_emb only every N batches; in-between batches reuse previous embedding")
    p.add_argument("--lllite_blocks", default=None,
                   help="comma-list subset of {down_blocks,mid_block,up_blocks}; default = all")
    p.add_argument("--lllite_proj", default=None,
                   help="comma-list subset of {attn1.to_q,attn1.to_k,attn1.to_v,attn2.to_q}; default = all")
    p.add_argument("--trt_engine", default=None,
                   help="path to compiled TensorRT engine; replaces pipeline.unet (skips torch.compile)")
    p.add_argument("--te_quant", choices=["fp16", "bf16", "nf4", "fp4", "int8"], default=None,
                   help="quantize Qwen3-VL TE for memory savings. nf4/fp4/int8 use bitsandbytes. "
                        "On 3090Ti class GPUs this loses speed but saves ~3x VRAM (deployment use case).")
    return p.parse_args()


def main():
    args = parse_args()

    # When TRT engine is provided, load the pipeline in fp16 to match the
    # engine's binding dtype.
    pipeline_dtype = torch.float16 if args.trt_engine else torch.bfloat16
    print(f"[load] {args.model} (dtype={pipeline_dtype})")
    pipeline = DreamLiteMobilePipeline.from_pretrained(
        args.model, torch_dtype=pipeline_dtype,
    ).to(args.device)

    if args.te_quant and args.te_quant not in ("fp16", "bf16"):
        from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        if args.te_quant == "int8":
            qconfig = BitsAndBytesConfig(load_in_8bit=True)
        else:
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=args.te_quant,
                bnb_4bit_use_double_quant=True,
            )
        te_path = Path(args.model) / "text_encoder"
        print(f"[te_quant] reloading TE from {te_path} as {args.te_quant}...")
        te_q = Qwen3VLForConditionalGeneration.from_pretrained(
            te_path, quantization_config=qconfig,
            device_map={"": args.device},
            torch_dtype=torch.float16,
        )
        del pipeline.text_encoder
        torch.cuda.empty_cache()
        pipeline.text_encoder = te_q
        n_params = sum(p.numel() for p in te_q.parameters())
        vram_gb = sum(p.element_size() * p.numel() for p in te_q.parameters()) / 1024**3
        print(f"  TE: {n_params/1e9:.2f}B params, VRAM ~{vram_gb:.2f} GB")

    if args.trt_engine:
        if args.lllite_weights:
            raise SystemExit(
                "TRT engine + LLLite are not yet compatible: the engine was "
                "built from base UNet so monkey-patched LLLite hooks are not "
                "in the compiled graph. Use either --trt_engine OR --lllite_weights."
            )
        from dreamlite_stream.trt_unet import TRTUNetWrapper
        print(f"[trt] swapping UNet for TRT engine: {args.trt_engine}")
        pipeline.unet = TRTUNetWrapper(args.trt_engine, device=args.device)
        if args.compile:
            print("[trt] disabling --compile (TRT already optimizes)")
            args.compile = False

    controller = None
    if args.lllite_weights:
        from dreamlite_lllite import apply_lllite
        from safetensors.torch import load_file
        vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
        latent_hw = args.size // vae_downsample
        block_filter = [s.strip() for s in args.lllite_blocks.split(",")] if args.lllite_blocks else None
        proj_filter = [s.strip() for s in args.lllite_proj.split(",")] if args.lllite_proj else None
        controller = apply_lllite(
            pipeline.unet, cond_emb_dim=32, mlp_dim=64,
            cond_image_size=args.size, sample_size=latent_hw,
            inference_mode=True, max_batch_size=args.batch_size,
            block_filter=block_filter, proj_filter=proj_filter,
        )
        # strict=False lets us load full-checkpoint weights even when we have
        # filtered down to a subset of hooks at inference.
        sd = load_file(args.lllite_weights)
        missing, unexpected = controller.load_state_dict(sd, strict=False)
        n_attached = len(controller.modules_dict)
        print(
            f"[lllite] loaded {n_attached} hooks  "
            f"(blocks={args.lllite_blocks or 'all'}  proj={args.lllite_proj or 'all'})  "
            f"multiplier={args.lllite_multiplier}  max_batch={args.batch_size}"
        )
        if unexpected:
            print(f"  (dropped {len(unexpected)} unused weight rows from checkpoint)")
        controller.to(device=args.device, dtype=torch.bfloat16)
        controller.eval()
        controller.set_multiplier(args.lllite_multiplier)

    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=args.steps, prompt=args.prompt,
    )
    worker = BatchedEditWorker(
        pipeline=pipeline, state=state, batch_size=args.batch_size,
        device=args.device, dtype=pipeline_dtype, seed=args.seed,
        compile=args.compile, compile_mode=args.compile_mode,
        lllite_controller=controller,
        fixed_noise=args.fixed_noise,
        cond_refresh_every=args.cond_refresh_every,
    )

    writer = None
    timings = []
    n_total = 0
    print(
        f"[run]  size={args.size}x{args.size} steps={args.steps} "
        f"batch={args.batch_size} compile={args.compile} pipelined={args.pipelined}"
    )
    t_start = time.perf_counter()
    batch_idx = 0
    fps_global = 24.0

    def write_outputs(outputs):
        nonlocal writer
        if writer is None:
            writer = VideoWriter(args.output, args.size, fps_global)
        for img in outputs:
            writer.write_pil(img)

    def log_batch(t, n):
        nonlocal batch_idx
        if batch_idx >= args.warmup:
            timings.append(t)
        print(
            f"[batch {batch_idx:03d}] {n} frames in "
            f"{t.total_ms:.0f}ms  ({t.per_frame_ms:.0f}ms/frame)  "
            f"te={t.te_ms:.0f}  enc={t.vae_enc_ms:.0f}  "
            f"den={t.denoise_ms:.0f}  dec={t.vae_dec_ms:.0f}",
            flush=True,
        )
        batch_idx += 1

    def collect_batch(it):
        """Pull up to batch_size frames; return ([frames], more_remaining)."""
        nonlocal n_total
        buf = []
        for idx, frame, fps in it:
            if args.max_frames is not None and idx >= args.max_frames:
                return buf, False
            nonlocal_set_fps(fps)
            buf.append(frame)
            n_total += 1
            if len(buf) >= args.batch_size:
                return buf, True
        return buf, False

    fps_holder = [24.0]
    def nonlocal_set_fps(v):
        fps_holder[0] = v

    iterator = iter_video_frames(args.input, args.size)
    # Drop partial batches: shape-locked TRT engine and LLLite cond_emb buffers
    # are sized for exactly batch_size; a partial last batch would crash.
    if not args.pipelined:
        # Synchronous batched
        while True:
            buf, more = collect_batch(iterator)
            fps_global = fps_holder[0]
            if not buf or len(buf) < args.batch_size:
                break
            outputs, t = worker.step_batch(buf)
            write_outputs(outputs)
            log_batch(t, len(buf))
            if not more:
                break
    else:
        # Pipelined batched: prefetch NEXT batch while denoising CURRENT
        cur_buf, more = collect_batch(iterator)
        fps_global = fps_holder[0]
        if cur_buf and len(cur_buf) == args.batch_size:
            cur_pf = worker.prefetch_batch(cur_buf)
            while True:
                if more:
                    nxt_buf, more = collect_batch(iterator)
                    if len(nxt_buf) < args.batch_size:
                        nxt_buf = []  # drop partial
                else:
                    nxt_buf = []
                fps_global = fps_holder[0]

                nxt_pf = worker.prefetch_batch(nxt_buf) if nxt_buf else None

                outputs, t = worker.step_batch_with_prefetch(cur_buf, cur_pf)
                write_outputs(outputs)
                log_batch(t, len(cur_buf))

                if not nxt_buf:
                    break
                cur_buf = nxt_buf
                cur_pf = nxt_pf

    if writer:
        writer.close()
    wall = (time.perf_counter() - t_start) * 1000

    if not timings:
        print("(no measured batches — increase --max_frames or reduce --warmup)")
        return
    n_measured = sum(t.n_frames for t in timings)
    sum_total = sum(t.total_ms for t in timings)
    fps_step = n_measured / (sum_total / 1000)
    fps_wall = n_total / (wall / 1000)
    avg_te = sum(t.te_ms * t.n_frames for t in timings) / n_measured
    avg_den = sum(t.denoise_ms * t.n_frames for t in timings) / n_measured
    avg_pf = sum(t.per_frame_ms * t.n_frames for t in timings) / n_measured

    print(
        f"\n[summary]\n"
        f"  total_frames={n_total}  measured={n_measured} (warmup={args.warmup} batch)\n"
        f"  fps_wall={fps_wall:.2f} (incl. first batch compile)\n"
        f"  fps_step={fps_step:.2f} (excl. warmup batches)\n"
        f"  per-frame avg ms: total={avg_pf:.1f}  te={avg_te:.1f}  denoise={avg_den:.1f}"
    )


if __name__ == "__main__":
    main()
