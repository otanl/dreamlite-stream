"""#144 functional probe: torch.compile on the Qwen3-VL vision tower.

Uses ``dreamlite_stream.te_vit_compile`` to compile ONLY the ViT
submodule of the DreamLite pipeline's Qwen3-VL TE, then verifies:

1. ``find_vision_tower`` locates the tower on the real model.
2. Compiled TE forward output matches the eager baseline (fp16 tolerance).
3. Informal timing: eager vs compiled (after warmup), per compile mode.

Windows caveat: inductor needs a working C++ toolchain; if compilation
fails this probe reports the failure mode honestly -- that IS the cell
value for this environment.

Run (main dreamlite-stream env):
    python scripts/test_vit_compile_qwen3vl.py --modes default reduce-overhead
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "src"))

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "te_vit_compile", os.path.join(_HERE, os.pardir, "src",
                                   "dreamlite_stream", "te_vit_compile.py"))
te_vit_compile = _ilu.module_from_spec(_spec)
sys.modules["te_vit_compile"] = te_vit_compile
_spec.loader.exec_module(te_vit_compile)


PROMPT_TEMPLATE = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "{}<|im_end|>\n<|im_start|>assistant\n"
)


def build_inputs(processor, prompt):
    from PIL import Image
    import numpy as np
    rng = np.random.default_rng(0)
    img = Image.fromarray(
        rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8))
    txt = PROMPT_TEMPLATE.format(prompt)
    tk = processor(text=[txt], images=[img], padding="max_length",
                   max_length=512, truncation=True, return_tensors="pt")
    return tk.to("cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"F:/work/dreamlite/models/DreamLite-mobile")
    ap.add_argument("--modes", nargs="+", default=["default"])
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModel

    print("loading Qwen3-VL TE + processor ...", flush=True)
    processor = AutoProcessor.from_pretrained(
        os.path.join(args.model, "processor"))
    tk = None

    def fresh_model():
        m = AutoModel.from_pretrained(
            os.path.join(args.model, "text_encoder"),
            torch_dtype=torch.float16)
        m.to("cuda").eval()
        return m

    def te_forward(model):
        with torch.no_grad():
            out = model(
                input_ids=tk.input_ids,
                attention_mask=tk.attention_mask,
                pixel_values=tk.pixel_values,
                image_grid_thw=tk.image_grid_thw,
                output_hidden_states=True,
            )
        return out.hidden_states[-1]

    def timed(model, warmup=2, iters=5):
        for _ in range(warmup):
            te_forward(model)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            r = te_forward(model)
        torch.cuda.synchronize()
        return r, (time.time() - t0) / iters * 1e3

    model = fresh_model()
    tk = build_inputs(processor, "transfer this to oil painting style")

    tower = te_vit_compile.find_vision_tower(model)
    print(f"find_vision_tower -> {type(tower).__name__ if tower else None}",
          flush=True)
    if tower is None:
        print("VIT_COMPILE_FAIL (tower not found)", flush=True)
        return 1

    base_h, base_ms = timed(model)
    print(f"eager baseline: {base_ms:.0f} ms, hidden {tuple(base_h.shape)}",
          flush=True)

    failures = 0
    for mode in args.modes:
        print(f"\n=== compile mode: {mode} ===", flush=True)
        m = fresh_model()
        try:
            cfg = te_vit_compile.CompileConfig(mode=mode)
            ok = te_vit_compile.compile_vision_tower(m, cfg)
            assert ok, "compile_vision_tower returned False"
            t0 = time.time()
            _ = te_forward(m)  # triggers compilation
            print(f"first (compiling) call: {time.time()-t0:.1f}s", flush=True)
            comp_h, comp_ms = timed(m)
            close = torch.allclose(comp_h.float(), base_h.float(),
                                   atol=5e-2, rtol=5e-2)
            mean_d = (comp_h.float() - base_h.float()).abs().mean().item()
            print(f"compiled: {comp_ms:.0f} ms (eager {base_ms:.0f} ms), "
                  f"allclose={close}, mean|delta|={mean_d:.5f}", flush=True)
            if not close and mean_d > 0.1:
                failures += 1
                print("  NUMERIC DIVERGENCE beyond tolerance", flush=True)
        except Exception as e:
            failures += 1
            print(f"FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
        finally:
            del m
            torch.cuda.empty_cache()

    print(f"\nVIT_COMPILE_{'FAIL' if failures else 'PASS'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
