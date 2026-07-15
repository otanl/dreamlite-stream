"""R3 functional test: IDPruner fixed-K visual-token budget on Mobile-O.

task #158 (W3). Wires dreamlite-stream's `idpruner_select`
(pure-torch MMR selection) onto Mobile-O's visual-token path.

Insertion point: `LlavaMetaForCausalLM.visual()` returns the projected
image features (B, V, D) which `prepare_inputs_labels_for_multimodal`
splices verbatim into the LLM input sequence. Pruning that tensor to
(B, K, D) *before* the splice shortens the LLM forward directly — no
forward hooks and no fixed-K padding needed at this stage, because the
sequence is rebuilt per call. (Fixed-K padding only becomes load-bearing
under CUDA-graph capture, which this substrate does not use yet.)

Contrast with DreamLite/Qwen3-VL, where visual tokens are already
embedded in the LLM input and the hook must sit at the decoder entry —
this asymmetry is a substrate-axis observation for the paper (sec 4/5).

Run:
    .venv-mobileo/Scripts/python.exe test_idpruner_mobileo.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# dreamlite-stream's te_pruning is pure torch. Load the module file
# directly (importlib) to avoid the package __init__, which pulls the
# full worker stack (cv2 etc.) not installed in .venv-mobileo.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "te_pruning",
    r"F:\work\dreamlite-stream\src\dreamlite_stream\te_pruning.py")
_te_pruning = _ilu.module_from_spec(_spec)
sys.modules["te_pruning"] = _te_pruning  # dataclass needs the module registered
_spec.loader.exec_module(_te_pruning)
IDPrunerConfig = _te_pruning.IDPrunerConfig
idpruner_select = _te_pruning.idpruner_select
from mobileo.model.builder import load_pretrained_model  # noqa: E402
from streaming_recipe import (  # noqa: E402
    build_edit_inputs, te_forward, denoise_with_cond)


class PrunedVisual:
    """Wraps model.visual with IDPruner selection on the projected tokens."""

    def __init__(self, model, budget_k: int, lambda_balance: float = 0.7):
        self.orig_visual = model.visual  # bound method (class attr lookup)
        self.cfg = IDPrunerConfig(budget_k=budget_k,
                                  lambda_balance=lambda_balance)
        self.last_v = None
        self.last_k = None

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.orig_visual(pixel_values)        # (B, V, D)
        self.last_v = feats.shape[1]
        sel = idpruner_select(feats, self.cfg)        # (B, K)
        sel, _ = sel.sort(dim=-1)                     # keep spatial order
        self.last_k = sel.shape[1]
        idx = sel.unsqueeze(-1).expand(-1, -1, feats.size(-1))
        return torch.gather(feats, 1, idx)            # (B, K, D)


def main():
    from PIL import Image
    out_dir = "smoke_outputs/idpruner"
    os.makedirs(out_dir, exist_ok=True)

    print("loading model ...", flush=True)
    tokenizer, model, _ = load_pretrained_model("checkpoints/Mobile-O-0.5B")
    model.to("cuda:0")
    model.to(torch.bfloat16)

    image = Image.open("assets/cute_cat.png").convert("RGB")
    instruction = "make it look like an oil painting"
    input_ids, image_tensor = build_edit_inputs(
        tokenizer, model, image, instruction)

    def timed_te():
        torch.cuda.synchronize()
        t0 = time.time()
        cond = te_forward(model, input_ids, image_tensor)
        torch.cuda.synchronize()
        return cond, (time.time() - t0) * 1e3

    # Baseline (no pruning)
    cond_base, ms_base = timed_te()
    assert not torch.isnan(cond_base).any()
    print(f"baseline: cond shape {tuple(cond_base.shape)}, "
          f"TE {ms_base:.0f} ms", flush=True)
    img = denoise_with_cond(model, cond_base, num_inference_steps=4)[0]
    img.save(f"{out_dir}/edit_K_full.png")

    results = []
    for k in (128, 64, 32):
        wrapper = PrunedVisual(model, budget_k=k)
        model.visual = wrapper  # instance attr shadows the class method
        try:
            cond, ms = timed_te()
            assert not torch.isnan(cond).any(), f"NaN cond at K={k}"
            img = denoise_with_cond(model, cond, num_inference_steps=4)[0]
            img.save(f"{out_dir}/edit_K{k}.png")
            results.append((k, wrapper.last_v, wrapper.last_k, ms, "OK"))
            print(f"K={k}: V={wrapper.last_v} -> {wrapper.last_k} tokens, "
                  f"cond {tuple(cond.shape)}, TE {ms:.0f} ms, saved "
                  f"edit_K{k}.png", flush=True)
        except Exception as e:
            results.append((k, wrapper.last_v, wrapper.last_k, -1,
                            f"FAIL {type(e).__name__}: {str(e)[:80]}"))
            print(f"K={k}: FAIL {type(e).__name__}: {e}", flush=True)
        finally:
            del model.visual  # restore the class method

    print("\n=== R3 Mobile-O summary (informal, shared GPU) ===", flush=True)
    print(f"  baseline: V=full,        TE {ms_base:.0f} ms", flush=True)
    for k, v, kk, ms, status in results:
        ms_s = f"{ms:.0f} ms" if ms >= 0 else "--"
        print(f"  K={k}: V={v} -> {kk},  TE {ms_s}  {status}", flush=True)

    n_fail = sum(1 for r in results if r[4] != "OK")
    print(f"R3_MOBILEO_{'FAIL' if n_fail else 'PASS'}", flush=True)


if __name__ == "__main__":
    main()
