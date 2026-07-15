"""LCM-LoRA distillation for DreamLite-mobile UNet.

Trains a small LoRA on UNet attention QKV linears to make the model
produce x_0-quality output in a single denoise step from pure noise.

Approach (latent-consistency-model style):
  - For each batch, sample two timesteps t > t' on the same trajectory.
  - Compute model predictions at both, propagate t -> t' with a single
    Euler step using the model's prediction.
  - Loss: ||model(x_t, t) - stop_grad(model(x_t', t'))||^2 (consistency)
  - + reconstruction term against teacher (4-step) sample for stability.

The trained LoRA is loaded at inference into the same EditWorker /
BatchedEditWorker for free-quality 1-step inference.

Note: this is the BASE UNet LoRA (no LLLite). For LLLite + LCM, train
this LoRA first, then re-train LLLite with the LCM-LoRA active.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402

from dreamlite_stream import pipeline_ops as ops  # noqa: E402
from dreamlite_stream.distillation_losses import (  # noqa: E402
    DistillationLossConfig,
    compute_distillation_loss,
    preset_B3_baseline,
    preset_O1_perceptual,
    preset_O2_spectral,
    preset_O3_main,
)


# ---------------------------------------------------------------------------
# Tiny LoRA (no peft dep)
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps an nn.Linear with a low-rank residual A @ B."""

    def __init__(self, host: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.host = host
        self.rank = rank
        self.alpha = alpha
        for p in self.host.parameters():
            p.requires_grad = False
        in_dim = host.in_features
        out_dim = host.out_features
        self.A = nn.Parameter(torch.zeros(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        # B remains zero-init so an untrained LoRA contributes nothing.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.host(x)
        # cast to host dtype for the LoRA path
        residual = F.linear(F.linear(x.to(self.A.dtype), self.A), self.B)
        return h + (self.alpha / self.rank) * residual.to(h.dtype)


def attach_unet_lora(unet, rank: int = 16, alpha: float = 16.0) -> List[LoRALinear]:
    """Attach LoRA to all attention QKV+O Linear layers in the UNet."""
    target_suffixes = (".attn1.to_q", ".attn1.to_k", ".attn1.to_v", ".attn1.to_out.0",
                       ".attn2.to_q", ".attn2.to_k", ".attn2.to_v", ".attn2.to_out.0")
    attached: List[LoRALinear] = []
    for name, module in unet.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        # Build wrapper and patch in place via parent
        parent_name, attr = name.rsplit(".", 1)
        parent = unet.get_submodule(parent_name)
        wrapper = LoRALinear(module, rank=rank, alpha=alpha)
        setattr(parent, attr, wrapper)
        attached.append(wrapper)
    return attached


# ---------------------------------------------------------------------------
# Training data: (input image, prompt) -> teacher x_0 latent
# ---------------------------------------------------------------------------
class FramePromptDataset(Dataset):
    """Reads pairs from a temporal_pairs manifest but only uses (input, target)
    for teacher-student LCM training. The temporal cond is ignored."""

    def __init__(self, pairs_dir: str, manifest_path: str):
        self.pairs_dir = Path(pairs_dir)
        self.rows = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _to_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(rgb_uint8).permute(2, 0, 1).contiguous().float() / 255.0
        return t * 2.0 - 1.0

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        npz_path = self.pairs_dir / "pairs" / f"{row['stem']}.npz"
        d = np.load(npz_path)
        return {
            "input": self._to_tensor(d["input"]),
            "target": self._to_tensor(d["target"]),
            "input_pil": Image.fromarray(d["input"]),
            "prompt": row["prompt"],
        }


def collate(batch):
    return {
        "inputs": torch.stack([b["input"] for b in batch], 0),
        "targets": torch.stack([b["target"] for b in batch], 0),
        "input_pils": [b["input_pil"] for b in batch],
        "prompts": [b["prompt"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--pairs_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=8)
    p.add_argument(
        "--max_steps", type=int, default=0,
        help="If >0, cap total optimisation steps (overrides max_epochs). "
             "Useful for short dry-runs."
    )
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    # --- Distillation loss family (follow-up paper, see
    # `notes/followup_design.md` §3.1 and
    # `src/dreamlite_stream/distillation_losses.py`).  Default config
    # matches the LCM-LoRA B2/B3 baseline (pure MSE).
    p.add_argument(
        "--loss_preset",
        choices=("B3", "O1", "O2", "O3"),
        default="B3",
        help=(
            "Loss configuration preset. B3 = pure MSE (baseline); "
            "O1 = MSE + perceptual; O2 = MSE + spectral; "
            "O3 = MSE + perceptual + spectral (main proposed config)."
        ),
    )
    p.add_argument(
        "--alpha_pixel", type=float, default=None,
        help="Override alpha (pixel MSE coefficient). If unset, preset value is used.",
    )
    p.add_argument(
        "--beta_perc", type=float, default=None,
        help="Override beta (LPIPS perceptual coefficient).",
    )
    p.add_argument(
        "--gamma_spec", type=float, default=None,
        help="Override gamma (HF-FFT spectral coefficient).",
    )
    p.add_argument(
        "--spec_on_pixel", action="store_true",
        help="Compute HF-FFT loss on RGB pixels (default: latent).",
    )
    p.add_argument(
        "--heavy_term_every_k", type=int, default=0,
        help=(
            "Compute LPIPS / pixel-spectral every k-th training step. "
            "0 (default) = every step. Set higher to save compute."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device("cuda")
    weight_dtype = torch.bfloat16

    print(f"[load] {args.model}")
    pipe = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=weight_dtype).to(device)
    vae, te, unet = pipe.vae, pipe.text_encoder, pipe.unet
    for m in (vae, te, unet):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # Attach LoRA
    print(f"[lora] attaching rank={args.rank} alpha={args.alpha}")
    loras = attach_unet_lora(unet, rank=args.rank, alpha=args.alpha)
    n_trainable = sum(p.numel() for w in loras for p in w.parameters() if p.requires_grad)
    print(f"  attached {len(loras)} LoRA wrappers, {n_trainable/1e6:.2f}M trainable params")
    # Move LoRA params to device, fp32 for stability
    lora_params = []
    for w in loras:
        for p in (w.A, w.B):
            p.data = p.data.to(device=device, dtype=torch.float32)
            p.requires_grad = True
            lora_params.append(p)

    ds = FramePromptDataset(args.pairs_dir, str(Path(args.pairs_dir) / "manifest.jsonl"))
    print(f"[data] {len(ds)} pairs")
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate, drop_last=True, pin_memory=True,
    )
    steps_per_epoch = max(1, len(ds) // (args.batch_size * args.gradient_accumulation_steps))
    max_steps = steps_per_epoch * args.max_epochs
    if args.max_steps > 0:
        max_steps = args.max_steps
        print(f"[sched] {steps_per_epoch} opt-steps/epoch; --max_steps override -> {max_steps} steps")
    else:
        print(f"[sched] {steps_per_epoch} opt-steps/epoch x {args.max_epochs} = {max_steps} steps")

    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, betas=(0.9, 0.999))
    warmup = min(100, max(1, max_steps // 10))
    def _lr_lambda(s):
        return min(1.0, s / warmup) if s < warmup else 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    vae_scaling = vae.config.scaling_factor
    add_time_ids = torch.tensor([[args.size, args.size]] * args.batch_size,
                                device=device, dtype=weight_dtype)

    # ------------------------------------------------------------------
    # Resolve distillation loss config (follow-up paper §3.1).
    _preset_map = {
        "B3": preset_B3_baseline,
        "O1": preset_O1_perceptual,
        "O2": preset_O2_spectral,
        "O3": preset_O3_main,
    }
    loss_cfg: DistillationLossConfig = _preset_map[args.loss_preset]()
    if args.alpha_pixel is not None:
        loss_cfg.alpha_pixel = args.alpha_pixel
    if args.beta_perc is not None:
        loss_cfg.beta_perc = args.beta_perc
    if args.gamma_spec is not None:
        loss_cfg.gamma_spec = args.gamma_spec
    if args.spec_on_pixel:
        loss_cfg.spec_on_pixel = True
    if args.heavy_term_every_k > 0:
        loss_cfg.heavy_term_every_k_steps = args.heavy_term_every_k
    print(
        f"[loss] preset={args.loss_preset} "
        f"alpha_pixel={loss_cfg.alpha_pixel} beta_perc={loss_cfg.beta_perc} "
        f"gamma_spec={loss_cfg.gamma_spec} spec_on_pixel={loss_cfg.spec_on_pixel} "
        f"heavy_every={loss_cfg.heavy_term_every_k_steps} "
        f"needs_pixel_decode={loss_cfg.needs_pixel_decode}",
        flush=True,
    )

    step = 0
    micro = 0
    t_start = time.time()
    losses_window = []

    print("[train]")
    while step < max_steps:
        for batch in dl:
            inputs = batch["inputs"].to(device=device, dtype=weight_dtype)
            targets = batch["targets"].to(device=device, dtype=weight_dtype)
            input_pils = batch["input_pils"]
            prompts = batch["prompts"]
            B = inputs.shape[0]

            with torch.no_grad():
                input_latents = vae.encode(inputs).latents
                target_latents = vae.encode(targets).latents

                # [Edit]-mode TE per row
                pe_list, pm_list = [], []
                for i in range(B):
                    decorated = (
                        "[Edit]: A diptych with two side-by-side images of the same scene. "
                        f"Compared to the right side, the left one has {prompts[i]}"
                    )
                    pe, pm = pipe.encode_prompt(
                        mode="edit", prompts=[decorated], image=input_pils[i],
                        device=device, dtype=weight_dtype,
                    )
                    pe_list.append(pe); pm_list.append(pm)
                L = max(p.shape[1] for p in pe_list)
                D = pe_list[0].shape[2]
                prompt_embeds = torch.zeros(B, L, D, device=device, dtype=weight_dtype)
                prompt_mask = torch.zeros(B, L, device=device, dtype=pm_list[0].dtype)
                for i, (pe, pm) in enumerate(zip(pe_list, pm_list)):
                    n = pe.shape[1]
                    prompt_embeds[i, :n] = pe[0]
                    prompt_mask[i, :n] = pm[0]

            # 1-step training: sample t close to 1.0 (start of trajectory).
            # Goal: model(noise, t=1) should predict target latents.
            t_val = 1.0
            t_b = torch.full((B,), t_val, device=device, dtype=weight_dtype)
            noise = torch.randn_like(target_latents)
            x_t = noise  # at sigma=1.0, x_t = noise
            v_target = noise - target_latents  # rectified-flow velocity

            # Spatial-concat input
            model_input = torch.cat([x_t, input_latents], dim=3)

            # Forward UNet (LoRA active)
            with torch.amp.autocast("cuda", dtype=weight_dtype):
                v_pred = unet(
                    model_input,
                    timestep=(t_b * 1000.0).to(weight_dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    added_cond_kwargs={"time_ids": add_time_ids[:B]},
                    return_dict=False,
                )[0]
                v_pred = v_pred[..., : target_latents.shape[-1]]

                # Distillation loss (follow-up). For B3 the pixel
                # MSE term is the only active term and the behaviour
                # matches the original `F.mse_loss(v_pred, v_target)`
                # code path bit-for-bit. For O1/O2/O3 the perceptual /
                # spectral terms are added on top; pixel decoding is
                # only done when the active config requires it.
                pix_s, pix_t = None, None
                if loss_cfg.needs_pixel_decode:
                    # Recover the predicted x_0 latent at t=1 from v.
                    # v = noise - x_0  =>  x_0 = noise - v.
                    x0_pred_latent = noise - v_pred
                    pix_s = vae.decode(x0_pred_latent / vae_scaling).sample
                    with torch.no_grad():
                        pix_t = vae.decode(target_latents / vae_scaling).sample

                loss_out = compute_distillation_loss(
                    loss_cfg,
                    student_v_pred=v_pred,
                    target_v=v_target,
                    student_pixels=pix_s,
                    teacher_pixels=pix_t,
                    student_latents=v_pred if not loss_cfg.spec_on_pixel else None,
                    teacher_latents=v_target if not loss_cfg.spec_on_pixel else None,
                    step=step,
                )
                loss = loss_out.total / args.gradient_accumulation_steps

            loss.backward()
            losses_window.append(loss.item() * args.gradient_accumulation_steps)
            micro += 1

            if micro % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    avg = sum(losses_window) / max(1, len(losses_window))
                    losses_window.clear()
                    elapsed = time.time() - t_start
                    print(f"  step {step}/{max_steps}  loss={avg:.4f}  sec={elapsed:.0f}", flush=True)

                if step % args.save_every == 0 or step == max_steps:
                    out = out_dir / f"lcm_lora_step{step:06d}.pt"
                    sd = {f"lora_{i}.A": w.A.detach().cpu() for i, w in enumerate(loras)}
                    sd.update({f"lora_{i}.B": w.B.detach().cpu() for i, w in enumerate(loras)})
                    sd["meta"] = {
                        "rank": args.rank, "alpha": args.alpha,
                        "n_loras": len(loras), "step": step,
                    }
                    torch.save(sd, out)
                    print(f"  saved {out}")

                if step >= max_steps:
                    break
        if step >= max_steps:
            break

    print(f"[done] trained for {step} steps in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
