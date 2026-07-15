"""bonsai-image 1-bit substrate: smoke test + recipe-cell probes.

third substrate (stretch / appendix column). Resolves three
matrix cells with functional evidence (informal timings, shared GPU):

1. SMOKE -- does the 1-bit FLUX.2-Klein stack load and generate in our
   main env (torch 2.11 + gemlite 0.5.1 + hqq 0.2.8)?
2. R1/R2 stage split -- time the TE stage (Qwen3-4B HQQ via
   ``_encode_klein_qwen3_prompt``) separately from the denoise loop.
   bonsai's TE is text-only: in a fixed-prompt streaming setting the
   cond is computed ONCE, so R1 degenerates to "static" (an A2/TE-family
   axis observation: no per-frame visual input -> nothing to refresh).
3. R5 conflict cell -- the design doc hypothesised "conflict with
   gemlite 1-bit GEMM". gemlite quantizes LINEAR-layer weights while
   SageAttn replaces the attention-activation kernel; they may compose.
   We patch F.scaled_dot_product_attention with sage_sdpa for a
   seed-matched generation and compare outputs.

Loading goes through the vendor backend (the HF snapshot is NOT a
standard diffusers layout):
    F:/work/bonsai/Bonsai-Image-Demo/vendor/image-studio/backend_gpu

Run (main dreamlite-stream env):
    python scripts/bonsai_smoke_probe.py
"""
from __future__ import annotations

import argparse
import inspect
import io
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "src"))

BACKEND_GPU_PARENT = r"F:\work\bonsai\Bonsai-Image-Demo\vendor\image-studio"
SNAPSHOT = r"F:\work\bonsai\checkpoints\bonsai-1bit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt",
                    default="oil painting of a dog running through a garden, "
                            "thick impasto brushstrokes, vibrant colors")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out-dir", default="out/bonsai_probe")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sys.path.insert(0, BACKEND_GPU_PARENT)
    from backend_gpu.pipeline_gpu import GpuPipeline  # noqa: E402

    results = {}

    # ---- 1. SMOKE: load + prewarm + one T2I ------------------------------
    print("=== building bonsai 1-bit GpuPipeline ===", flush=True)
    t0 = time.time()
    pipe = GpuPipeline(
        backend="bonsai-binary-gemlite",
        binary_transformer_path=os.path.join(SNAPSHOT, "transformer-gemlite-int1"),
        # never loaded (we stay on the binary backend); required eagerly by
        # the path table, so point it at the same artefact.
        ternary_transformer_path=os.path.join(SNAPSHOT, "transformer-gemlite-int1"),
        text_encoder_path=os.path.join(SNAPSHOT, "text_encoder-hqq-4bit"),
        vae_path=os.path.join(SNAPSHOT, "vae"),
        tokenizer_path=os.path.join(SNAPSHOT, "text_encoder-hqq-4bit", "tokenizer"),
    )
    pipe.prewarm()
    print(f"prewarmed in {time.time()-t0:.1f}s", flush=True)

    def gen(seed=7):
        return pipe.generate_png(prompt=args.prompt, seed=seed,
                                 steps=args.steps, height=args.size,
                                 width=args.size, guidance=1.0)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.time()
    png = gen()
    torch.cuda.synchronize()
    dt = time.time() - t0
    with open(os.path.join(args.out_dir, "smoke_t2i.png"), "wb") as f:
        f.write(png)
    peak = torch.cuda.max_memory_allocated() / 1e9
    results["smoke"] = f"OK {dt:.1f}s/{args.size}^2 @{args.steps} steps, peak {peak:.1f} GB"
    print(f"[smoke] {results['smoke']}", flush=True)

    # warm reference
    torch.cuda.synchronize(); t0 = time.time()
    png_warm = gen()
    torch.cuda.synchronize()
    warm_s = time.time() - t0
    results["warm_full"] = f"{warm_s:.1f}s warm full call"
    print(f"[warm] {results['warm_full']}", flush=True)

    # ---- 2. TE stage split ------------------------------------------------
    print("\n=== TE stage (text-only Qwen3-4B HQQ) ===", flush=True)
    try:
        from backend_gpu import diffusion_klein
        enc = diffusion_klein._encode_klein_qwen3_prompt
        sig = inspect.signature(enc)
        kw = {}
        for name in sig.parameters:
            if name in ("tokenizer",):
                kw[name] = pipe._tokenizer
            elif name in ("text_encoder",):
                kw[name] = pipe._text_encoder
            elif name in ("prompt", "prompts"):
                kw[name] = args.prompt
            elif name == "max_sequence_length":
                kw[name] = 512
        # fill remaining required params with their defaults only
        torch.cuda.synchronize(); t0 = time.time()
        emb = enc(**kw)
        torch.cuda.synchronize()
        te_ms = (time.time() - t0) * 1e3
        shape = tuple(emb.shape) if hasattr(emb, "shape") else type(emb).__name__
        results["te_stage"] = (f"{te_ms:.0f} ms, embeds {shape} -- text-only; "
                               "static under fixed-prompt streaming (R1 "
                               "degenerates: nothing to refresh per frame)")
        print(f"[te] {results['te_stage']}", flush=True)
    except Exception as e:
        results["te_stage"] = f"introspection failed: {type(e).__name__}: {str(e)[:90]}"
        print(f"[te] {results['te_stage']}", flush=True)

    # ---- 3. R5 x gemlite composition test --------------------------------
    print("\n=== R5 x gemlite composition (sage_sdpa global patch) ===",
          flush=True)
    try:
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "sage_attn", os.path.join(_HERE, os.pardir, "src",
                                      "dreamlite_stream", "sage_attn.py"))
        sage_attn = ilu.module_from_spec(spec)
        sys.modules["sage_attn"] = sage_attn
        spec.loader.exec_module(sage_attn)

        ref_png = gen(seed=11)
        handle = sage_attn.install_global_sdpa_patch()
        try:
            torch.cuda.synchronize(); t0 = time.time()
            sage_png = gen(seed=11)
            torch.cuda.synchronize()
            sage_s = time.time() - t0
            stats = dict(sage_attn.PATCH_STATS)
        finally:
            handle.remove()
        print(f"  patch route stats: {stats}", flush=True)

        from PIL import Image
        import numpy as np
        a = np.asarray(Image.open(io.BytesIO(ref_png)).convert("RGB")).astype("float32")
        b = np.asarray(Image.open(io.BytesIO(sage_png)).convert("RGB")).astype("float32")
        mad = float(np.abs(a - b).mean())
        Image.open(io.BytesIO(ref_png)).save(os.path.join(args.out_dir, "r5_ref.png"))
        Image.open(io.BytesIO(sage_png)).save(os.path.join(args.out_dir, "r5_sage.png"))
        degenerate = bool(b.max() < 5.0) or bool(np.isnan(b).any())
        if stats.get("sage", 0) == 0:
            verdict = "UNTESTED (all attention sites fell back to SDPA)"
        elif mad < 25.0 and not degenerate:
            verdict = "COMPOSES"
        else:
            verdict = "DIVERGES"
        results["r5_gemlite"] = (f"{verdict}: mean-abs-diff {mad:.1f}/255, "
                                 f"routes {stats}, "
                                 f"sage-patched {sage_s:.1f}s vs warm {warm_s:.1f}s")
        print(f"[r5] {results['r5_gemlite']}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        results["r5_gemlite"] = f"FAIL {type(e).__name__}: {str(e)[:100]}"
        print(f"[r5] {results['r5_gemlite']}", flush=True)

    # ---- Summary ----------------------------------------------------------
    print("\n=== BONSAI PROBE SUMMARY ===", flush=True)
    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)
    n_bad = sum(1 for v in results.values() if v.startswith("FAIL"))
    print(f"BONSAI_PROBE_{'PARTIAL' if n_bad else 'PASS'}", flush=True)


if __name__ == "__main__":
    main()
