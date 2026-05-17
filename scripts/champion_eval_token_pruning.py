"""Champion eval with system prompt removed from the TE template.

Compares quality metrics (warp_err, consistency, fps) against the
original template. Quality regression decides whether the 12% TE
speedup is worth taking.

Run twice: --template full (baseline) and --template no_system.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from statistics import mean, stdev

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
from dreamlite_lllite import apply_lllite  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from dreamlite_stream import BatchedEditWorker, SharedState  # noqa: E402
from dreamlite_stream import pipeline_ops as ops  # noqa: E402
from dreamlite_stream.metrics import compute_temporal, read_video_frames  # noqa: E402
from dreamlite_stream.runtime import VideoWriter, iter_video_frames  # noqa: E402


# ------------------------------------------------------------------
# Template variants
# ------------------------------------------------------------------
TEMPLATE_FULL = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text instruction should alter "
    "or modify the image. Generate a new image that meets the user's requirements while maintaining "
    "consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
)
TEMPLATE_NO_SYSTEM = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
)
TEMPLATE_TINY = (
    "<|im_start|>system\nEdit the image.<|im_end|>\n<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
)


def _patch_encode_prompt(template: str, drop_idx: int):
    """Monkey-patch ops.encode_prompt_edit_batch to use a given template
    and drop_idx. Returns a tuple of (original_fn, restore_fn) for cleanup.
    """
    from torch.nn.utils.rnn import pad_sequence

    orig = ops.encode_prompt_edit_batch

    @torch.no_grad()
    def _patched(pipeline, prompts, images, device, dtype):
        if len(prompts) != len(images):
            raise ValueError("mismatched lengths")
        from PIL import Image
        decorated = [
            f"[Edit]: A diptych with two side-by-side images of the same scene. "
            f"Compared to the right side, the left one has {p}"
            for p in prompts
        ]
        txts = [template.format(p) for p in decorated]
        pil_imgs = [img.resize((256, 256), Image.Resampling.LANCZOS) for img in images]
        tk_out = pipeline.processor(
            text=txts, images=pil_imgs, padding=True, return_tensors="pt",
        ).to(device)
        outputs = pipeline.text_encoder(
            input_ids=tk_out.input_ids,
            attention_mask=tk_out.attention_mask,
            pixel_values=tk_out.pixel_values,
            image_grid_thw=tk_out.image_grid_thw,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        split = pipeline._extract_masked_hidden(hidden_states, tk_out.attention_mask)
        split = [e[drop_idx:] for e in split]
        prompt_embeds = pad_sequence(split, batch_first=True, padding_value=0).to(
            dtype=dtype, device=device,
        )
        B, L, _ = prompt_embeds.shape
        prompt_embeds_mask = torch.zeros((B, L), dtype=torch.long, device=device)
        for i, seq in enumerate(split):
            prompt_embeds_mask[i, : seq.shape[0]] = 1
        return prompt_embeds, prompt_embeds_mask

    ops.encode_prompt_edit_batch = _patched
    return orig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights",
                   default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--mp4_dir", default=str(_ROOT / "assets" / "davis_mp4"))
    p.add_argument("--sequences", nargs="+", default=[
        "blackswan", "libby", "swing", "camel", "dance-twirl",
    ])
    p.add_argument("--template", choices=["full", "no_system", "tiny"], default="full")
    p.add_argument("--drop_idx", type=int, default=None,
                   help="if None, picks a sensible default per template")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cond_refresh_every", type=int, default=8)
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out_dir", default=str(_ROOT / "out" / "token_pruning"))
    return p.parse_args()


@torch.no_grad()
def run_sequence(pipeline, controller, args, seq_name, out_dir):
    in_path = Path(args.mp4_dir) / f"{seq_name}.mp4"
    if not in_path.exists():
        return None
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
    out_path = out_dir / args.template / f"{seq_name}.mp4"
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
        "sequence": seq_name,
        "fps": fps_step,
        "n_frames": n_total,
        "out_path": str(out_path),
        "te_ms_mean": mean(t.te_ms for t in timings),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick template + drop_idx defaults
    if args.template == "full":
        template = TEMPLATE_FULL
        default_drop = 64
    elif args.template == "no_system":
        template = TEMPLATE_NO_SYSTEM
        default_drop = 4   # `<|im_start|>user\n` is ~4 tokens
    else:  # tiny
        template = TEMPLATE_TINY
        default_drop = 10
    drop_idx = args.drop_idx if args.drop_idx is not None else default_drop
    print(f"[template] {args.template}, drop_idx={drop_idx}")
    _patch_encode_prompt(template, drop_idx)

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
    print(f"[lllite] {len(controller.modules_dict)} hooks")

    rows = []
    for seq in args.sequences:
        print(f"\n=== {seq} ({args.template}) ===")
        t0 = time.perf_counter()
        result = run_sequence(pipeline, controller, args, seq, out_dir)
        if result is None:
            continue
        elapsed = time.perf_counter() - t0
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
            f"  fps={result['fps']:5.2f}  te_ms={result['te_ms_mean']:5.1f}  "
            f"warp_err={m.warping_error:5.2f}  con_l1={m.consecutive_l1:5.2f}  "
            f"ratio={m.consistency_ratio:.3f}  ({elapsed:.0f}s)"
        )

    if rows:
        fpss = [r["fps"] for r in rows]
        wes = [r["warp_err"] for r in rows]
        tes = [r["te_ms_mean"] for r in rows]
        print()
        print("=" * 70)
        print(f"template={args.template}")
        print(
            f"aggregate: fps {mean(fpss):.2f} ± {stdev(fpss) if len(fpss)>1 else 0:.2f}  "
            f"te_ms {mean(tes):.1f}  warp_err {mean(wes):.2f} ± {stdev(wes) if len(wes)>1 else 0:.2f}  "
            f"N={len(rows)}"
        )

        results_path = out_dir / f"results_{args.template}.jsonl"
        import json
        with open(results_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[saved] {results_path}")


if __name__ == "__main__":
    main()
