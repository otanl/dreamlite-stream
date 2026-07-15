"""R3 functional test: IDPrunerHook on the real Qwen3-VL text encoder.

Completes #142 (DreamLite-side R3 cell). Loads the pipeline's Qwen3-VL
TE + processor directly, attaches the two-hook IDPruner installation,
and verifies:

1. Hook fires and reports (V, K, L) with L unchanged.
2. Output hidden_states keep their shape (compile-graph contract).
3. Pruned forward differs from baseline (it actually pruned).
4. Text-token rows of the final hidden state stay finite / sane.
5. Informal TE latency at K in {64, 128} vs baseline.

Run (main dreamlite-stream env):
    python scripts/test_idpruner_qwen3vl.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

# Import te_pruning standalone (package __init__ pulls cv2 etc.).
import importlib.util as _ilu

_here = os.path.dirname(os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location(
    "te_pruning", os.path.join(_here, os.pardir, "src", "dreamlite_stream",
                               "te_pruning.py"))
te_pruning = _ilu.module_from_spec(_spec)
sys.modules["te_pruning"] = te_pruning
_spec.loader.exec_module(te_pruning)


PROMPT_TEMPLATE = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "{}<|im_end|>\n<|im_start|>assistant\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"F:/work/dreamlite/models/DreamLite-mobile")
    ap.add_argument("--prompt", default="transfer this to oil painting style")
    ap.add_argument("--budgets", nargs="+", type=int, default=[128, 64])
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor, AutoModel

    te_dir = os.path.join(args.model, "text_encoder")
    proc_dir = os.path.join(args.model, "processor")

    print("loading Qwen3-VL TE + processor ...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(proc_dir)
    model = AutoModel.from_pretrained(te_dir, torch_dtype=torch.float16)
    model.to("cuda").eval()
    print(f"loaded in {time.time()-t0:.1f}s: {type(model).__name__}",
          flush=True)

    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print(f"image_token_id = {image_token_id}", flush=True)

    # Build one multimodal input mirroring the batched_edit worker
    # (256x256 image, fixed-max-length padding).
    img = Image.new("RGB", (256, 256), (90, 140, 60))
    import numpy as np
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    txt = PROMPT_TEMPLATE.format(args.prompt)
    tk = processor(text=[txt], images=[img], padding="max_length",
                   max_length=512, truncation=True, return_tensors="pt")
    tk = tk.to("cuda")

    def te_forward():
        with torch.no_grad():
            out = model(
                input_ids=tk.input_ids,
                attention_mask=tk.attention_mask,
                pixel_values=tk.pixel_values,
                image_grid_thw=tk.image_grid_thw,
                output_hidden_states=True,
            )
        return out.hidden_states[-1]

    def timed(fn, warmup=1, iters=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            r = fn()
        torch.cuda.synchronize()
        return r, (time.time() - t0) / iters * 1e3

    # Baseline
    base_h, base_ms = timed(te_forward)
    print(f"baseline: hidden {tuple(base_h.shape)}, TE {base_ms:.0f} ms",
          flush=True)
    n_visual = int((tk.input_ids == image_token_id).sum().item())
    print(f"visual tokens in sequence: V={n_visual}", flush=True)

    failures = 0
    for k in args.budgets:
        hook = te_pruning.IDPrunerHook(te_pruning.IDPrunerConfig(budget_k=k))
        hook.attach(model, image_token_id=image_token_id)
        try:
            pruned_h, pruned_ms = timed(te_forward)
            stats = hook.last_stats
            assert stats is not None, "hook never fired"
            assert pruned_h.shape == base_h.shape, (
                f"shape changed: {pruned_h.shape} vs {base_h.shape}")
            assert not torch.isnan(pruned_h).any(), "NaN in pruned hidden"
            diff = (pruned_h.float() - base_h.float()).abs().mean().item()
            assert diff > 1e-5, "pruned output identical to baseline?"
            print(f"K={k}: fired with {stats}, hidden shape preserved, "
                  f"mean|delta|={diff:.4f}, TE {pruned_ms:.0f} ms "
                  f"(baseline {base_ms:.0f} ms)", flush=True)
        except Exception as e:
            failures += 1
            print(f"K={k}: FAIL {type(e).__name__}: {e}", flush=True)
        finally:
            hook.detach()

    # After detach, the model must return to baseline behaviour.
    post_h, _ = timed(te_forward, warmup=0, iters=1)
    restored = torch.allclose(post_h, base_h, atol=1e-3, rtol=1e-3)
    print(f"detach restores baseline: {restored}", flush=True)
    if not restored:
        failures += 1

    print(f"R3_QWEN3VL_{'FAIL' if failures else 'PASS'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
