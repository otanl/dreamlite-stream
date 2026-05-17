"""Export DreamLite UNet WITH LLLite hooks baked in to ONNX.

Each LLLite hook's `cond_emb` becomes a separate ONNX graph input. The wrapper
forward takes (model_input, timestep, ehs, eam, time_ids, *cond_embs).

Trace strategy:
  Before tracing UNet, we set `m._export_cond_emb = cond_embs[i]` on each
  LLLite module. Each hook's forward (replaced with `forward_export`) reads
  `self._export_cond_emb` — Python attribute load is not traced, but the
  underlying tensor (a graph input) IS the value the tracer captures.

Limitation: number of hooks and their order are fixed at export time. Run
this with the same `--lllite_blocks` you intend to use at inference.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from types import MethodType

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights",
                   default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--out", default=str(_ROOT / "out" / "trt" / "unet_lllite_b8_512.onnx"))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--prompt_seq_len", type=int, default=200)
    p.add_argument("--encoder_hid_dim", type=int, default=2048)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    return p.parse_args()


def forward_export(self, x):
    """LLLite-hook forward that reads cond_emb from an attribute set by the
    outer wrapper. This is what host.forward gets replaced with for export.
    """
    cx = self._export_cond_emb
    d = self.down(x)
    merged = torch.cat([cx, d], dim=2)
    merged = self.mid(merged)
    delta = self.up(merged) * self._multiplier_t
    return self._org_forward(x + delta)


class UNetWithLLLite(torch.nn.Module):
    """Wrapper that exposes (model_input, timestep, ehs, eam, time_ids, *cond_embs)
    as positional inputs. Each cond_emb is routed to a specific LLLite hook
    by ordering: cond_embs[i] -> hook_names[i].
    """

    def __init__(self, unet, controller, hook_names):
        super().__init__()
        self.unet = unet
        self.controller = controller
        self.hook_names = list(hook_names)
        # Keep references to the actual LLLite modules in order
        self._modules_in_order = [controller.modules_dict[n] for n in self.hook_names]

    def forward(self, model_input, timestep, encoder_hidden_states,
                encoder_attention_mask, time_ids, *cond_embs):
        assert len(cond_embs) == len(self._modules_in_order), \
            f"Expected {len(self._modules_in_order)} cond_embs, got {len(cond_embs)}"
        # Set each hook's _export_cond_emb to the corresponding input tensor
        for m, ce in zip(self._modules_in_order, cond_embs):
            m._export_cond_emb = ce
        return self.unet(
            model_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            added_cond_kwargs={"time_ids": time_ids},
            return_dict=False,
        )[0]


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")
    B = args.batch_size

    print(f"[load] {args.model} (dtype={args.dtype})")
    pipeline = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    unet = pipeline.unet
    unet.eval()

    # Apply LLLite (same as champion_eval)
    vae_scale = pipeline.vae_scale_factor
    latent_hw = args.size // vae_scale
    block_filter = [s.strip() for s in args.lllite_blocks.split(",")] if args.lllite_blocks else None
    controller = apply_lllite(
        unet, cond_emb_dim=32, mlp_dim=64,
        cond_image_size=args.size, sample_size=latent_hw,
        inference_mode=False,   # We'll override host.forward ourselves below
        max_batch_size=B,
        block_filter=block_filter,
    )
    sd = load_file(args.lllite_weights)
    controller.load_state_dict(sd, strict=False)
    controller.to(device=device, dtype=dtype)
    controller.eval()
    controller.set_multiplier(1.0)

    n_hooks = len(controller.modules_dict)
    print(f"[lllite] {n_hooks} hooks attached (blocks={args.lllite_blocks})")

    # Swap each hook's host.forward to our forward_export
    hook_names = list(controller.modules_dict.keys())
    for name in hook_names:
        m = controller.modules_dict[name]
        host = m._host_ref[0]
        # forward_export needs (x) only; cond_emb comes from m._export_cond_emb
        host.forward = MethodType(forward_export, m)

    # Sample inputs
    model_input = torch.randn(B, 4, latent_hw, latent_hw * 2, device=device, dtype=dtype)
    timestep = torch.full((B,), 500.0, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(B, args.prompt_seq_len, args.encoder_hid_dim, device=device, dtype=dtype)
    encoder_attention_mask = torch.ones(B, args.prompt_seq_len, device=device, dtype=torch.long)
    time_ids = torch.tensor([[args.size, args.size]] * B, device=device, dtype=dtype)

    # Build sample cond_embs (one per hook). Shape: (B, seq_len, cond_emb_dim).
    # seq_len = block_feature_size^2 * width_concat_factor. We can pull this
    # from each module's _cond_emb_buf shape.
    cond_embs = []
    for name in hook_names:
        m = controller.modules_dict[name]
        buf = m._cond_emb_buf
        ce = torch.randn(B, buf.shape[1], buf.shape[2], device=device, dtype=dtype)
        cond_embs.append(ce)

    wrapper = UNetWithLLLite(unet, controller, hook_names).to(device).eval()

    print(f"[shapes]")
    print(f"  model_input         : {tuple(model_input.shape)} {model_input.dtype}")
    print(f"  timestep            : {tuple(timestep.shape)}")
    print(f"  encoder_hidden      : {tuple(encoder_hidden_states.shape)}")
    print(f"  encoder_attn_mask   : {tuple(encoder_attention_mask.shape)}")
    print(f"  time_ids            : {tuple(time_ids.shape)}")
    print(f"  cond_embs           : {len(cond_embs)} tensors, first shape {tuple(cond_embs[0].shape)}")

    # Sanity check forward
    print("[sanity] running PyTorch wrapper forward...")
    with torch.no_grad():
        ref_out = wrapper(model_input, timestep, encoder_hidden_states,
                          encoder_attention_mask, time_ids, *cond_embs)
    print(f"  ref_out.shape = {tuple(ref_out.shape)}, dtype={ref_out.dtype}")

    # Build input_names list
    input_names = ["model_input", "timestep", "encoder_hidden_states",
                   "encoder_attention_mask", "time_ids"]
    for i in range(len(cond_embs)):
        input_names.append(f"cond_emb_{i}")

    print(f"\n[export] writing {out_path}  (input count: {len(input_names)})")
    inputs_tuple = (
        model_input, timestep, encoder_hidden_states,
        encoder_attention_mask, time_ids,
        *cond_embs,
    )
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs_tuple,
            str(out_path),
            input_names=input_names,
            output_names=["noise_pred"],
            opset_version=17,
            do_constant_folding=True,
        )

    print(f"[done] saved ONNX: {out_path}")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  file size: {size_mb:.1f} MB")

    # Also dump the hook->binding mapping for the runtime wrapper
    mapping_path = out_path.with_suffix(".hooks.json")
    import json
    with open(mapping_path, "w") as f:
        json.dump({
            "n_hooks": n_hooks,
            "hook_names": hook_names,
            "lllite_blocks": args.lllite_blocks,
            "batch_size": B,
            "size": args.size,
            "cond_emb_shape": list(cond_embs[0].shape),
        }, f, indent=2)
    print(f"  wrote hook mapping: {mapping_path}")


if __name__ == "__main__":
    main()
