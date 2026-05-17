"""Train a temporal LLLite adapter for DreamLite-mobile.

Adapts dreamlite-lllite/scripts/train_lllite.py for our frame-pair data:

  - Mode is [Edit], not [Generate]:
      model_input = cat([x_t, input_latent], dim=3)
      prompt_embeds = encode_prompt(mode="edit", image=input_PIL, ...)
  - LLLite cond_image is the warped previous (blended) target —
    the LLLite learns to bridge "warped prev consistent target" -> "next
    consistent target" given the new input frame.
  - Loss is the standard rectified-flow / MSE on the velocity prediction.

Usage:
    python scripts/train_temporal_lllite.py \
        --pairs_dir data/temporal_pairs_distill \
        --out_dir runs/temporal_lllite_v1 \
        --size 512 --batch_size 1 --gradient_accumulation_steps 4 \
        --max_epochs 12
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
_LLLITE = _ROOT.parent / "dreamlite-lllite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))
sys.path.insert(0, str(_LLLITE / "src"))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402
from dreamlite_lllite import apply_lllite  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TemporalPairDataset(Dataset):
    """Reads pairs/*.npz produced by generate_temporal_pairs.py.

    Each item:
        input        (3, H, W) float in [-1, 1]   — VAE input for cond_image_latents
        cond         (3, H, W) float in [-1, 1]   — warped prev target (LLLite cond)
        target       (3, H, W) float in [-1, 1]   — VAE input for the noise-prediction target
        input_pil    PIL.Image                    — for [Edit]-mode TE (vision tokens)
        prompt       str
    """

    def __init__(self, pairs_dir: str, manifest_path: str):
        self.pairs_dir = Path(pairs_dir)
        rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _to_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
        # (H, W, 3) uint8 -> (3, H, W) float in [-1, 1]
        t = torch.from_numpy(rgb_uint8).permute(2, 0, 1).contiguous().float() / 255.0
        return t * 2.0 - 1.0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        npz_path = self.pairs_dir / "pairs" / f"{row['stem']}.npz"
        d = np.load(npz_path)
        in_rgb = d["input"]
        wprev_rgb = d["warped_target_prev"]
        tgt_rgb = d["target"]
        return {
            "input":     self._to_tensor(in_rgb),
            "cond":      self._to_tensor(wprev_rgb),
            "target":    self._to_tensor(tgt_rgb),
            "input_pil": Image.fromarray(in_rgb),
            "prompt":    row["prompt"],
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "inputs":     torch.stack([b["input"] for b in batch], 0),
        "conds":      torch.stack([b["cond"] for b in batch], 0),
        "targets":    torch.stack([b["target"] for b in batch], 0),
        "input_pils": [b["input_pil"] for b in batch],
        "prompts":    [b["prompt"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _freeze(*modules: torch.nn.Module) -> None:
    for m in modules:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    ap.add_argument("--pairs_dir", required=True,
                    help="root dir produced by generate_temporal_pairs.py "
                         "(must contain pairs/ and manifest.jsonl)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--max_epochs", type=int, default=12)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--cond_emb_dim", type=int, default=32)
    ap.add_argument("--mlp_dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--mixed_precision", choices=["bf16", "no"], default="bf16")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    _seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device("cuda")
    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    # -- Pipeline + freeze
    print(f"[load] {args.model}")
    pipe = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=weight_dtype).to(device)
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    _freeze(vae, text_encoder, unet)

    if args.gradient_checkpointing:
        for m in unet.modules():
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = True
        unet.train()

    # -- LLLite
    vae_downsample = 2 ** (len(vae.config.encoder_block_out_channels) - 1)
    latent_hw = args.size // vae_downsample
    controller = apply_lllite(
        unet,
        cond_emb_dim=args.cond_emb_dim,
        mlp_dim=args.mlp_dim,
        cond_image_size=args.size,
        sample_size=latent_hw,
    )
    controller.to(device=device, dtype=torch.float32)
    controller.train()
    n_train = controller.num_parameters()
    print(f"[lllite] trainable params: {n_train:,} ({n_train/1e6:.2f} M)")

    # -- Dataset
    manifest_path = Path(args.pairs_dir) / "manifest.jsonl"
    ds = TemporalPairDataset(args.pairs_dir, str(manifest_path))
    print(f"[data] {len(ds)} pairs")
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_collate,
        drop_last=True, pin_memory=True,
    )
    steps_per_epoch = max(1, len(ds) // (args.batch_size * args.gradient_accumulation_steps))
    max_steps = steps_per_epoch * args.max_epochs
    print(f"[sched] {steps_per_epoch} opt-steps/epoch x {args.max_epochs} epochs = {max_steps} steps")

    # -- Optimizer
    optimizer = torch.optim.AdamW(
        controller.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8,
    )
    warmup = min(100, max(1, max_steps // 10))

    def _lr_lambda(s: int) -> float:
        return min(1.0, s / warmup) if s < warmup else 1.0
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # -- Constants
    vae_scaling = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0.0)
    add_time_ids = torch.tensor(
        [[args.size, args.size]] * args.batch_size,
        device=device, dtype=weight_dtype,
    )

    # -- Training loop
    step = 0
    micro = 0
    t_start = time.time()
    pbar = tqdm(total=max_steps, desc="train")
    losses_window: List[float] = []

    while step < max_steps:
        for batch in dl:
            inputs   = batch["inputs"].to(device=device, dtype=weight_dtype)   # (B, 3, H, W)
            conds    = batch["conds"].to(device=device, dtype=weight_dtype)
            targets  = batch["targets"].to(device=device, dtype=weight_dtype)
            input_pils = batch["input_pils"]
            prompts  = batch["prompts"]
            B = inputs.shape[0]

            with torch.no_grad():
                # VAE encode target (= ground-truth latents) and input (=
                # cond_image_latents for the spatial-concat right half).
                latents       = vae.encode(targets).latents
                input_latents = vae.encode(inputs).latents
                # DreamLite uses scaling_factor only on UNet input/output;
                # vae.encode().latents are already in the "raw" latent space.
                # Mirror inference pipeline: we don't scale here (set_timesteps
                # treats them as is), matching pipeline_dreamlite_mobile.

                # [Edit]-mode TE — image-dependent. One sample per batch row.
                prompt_embeds_list = []
                prompt_mask_list = []
                for i in range(B):
                    decorated = (
                        "[Edit]: A diptych with two side-by-side images of the same scene. "
                        f"Compared to the right side, the left one has {prompts[i]}"
                    )
                    pe, pm = pipe.encode_prompt(
                        mode="edit",
                        prompts=[decorated],
                        image=input_pils[i],
                        device=device,
                        dtype=weight_dtype,
                    )
                    prompt_embeds_list.append(pe)
                    prompt_mask_list.append(pm)
                # Pad to common length
                L = max(p.shape[1] for p in prompt_embeds_list)
                D = prompt_embeds_list[0].shape[2]
                prompt_embeds = torch.zeros(B, L, D, device=device, dtype=weight_dtype)
                prompt_mask   = torch.zeros(B, L, device=device, dtype=prompt_mask_list[0].dtype)
                for i, (pe, pm) in enumerate(zip(prompt_embeds_list, prompt_mask_list)):
                    n = pe.shape[1]
                    prompt_embeds[i, :n] = pe[0]
                    prompt_mask[i, :n]   = pm[0]

            # Set LLLite cond image (precompute embeds once per batch)
            controller.set_cond_image(conds)

            # Sample t (logit-normal, like FLUX/SD3 training)
            u = torch.randn(B, device=device)
            t = torch.sigmoid(u).to(weight_dtype)
            noise = torch.randn_like(latents)
            t_b = t.view(B, 1, 1, 1)
            x_t = (1 - t_b) * latents + t_b * noise
            v_target = noise - latents

            # Spatial-concat input for [Edit] mode: right half = input latents
            model_input = torch.cat([x_t, input_latents], dim=3)

            with torch.amp.autocast("cuda", dtype=weight_dtype, enabled=args.mixed_precision == "bf16"):
                v_pred = unet(
                    model_input,
                    timestep=(t * 1000.0).to(weight_dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    added_cond_kwargs={"time_ids": add_time_ids[:B]},
                    return_dict=False,
                )[0]
                v_pred = v_pred[..., : latents.shape[-1]]
                loss = F.mse_loss(v_pred.float(), v_target.float()) / args.gradient_accumulation_steps

            loss.backward()
            losses_window.append(loss.item() * args.gradient_accumulation_steps)
            micro += 1

            if micro % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)

                if step % args.log_every == 0:
                    avg = sum(losses_window) / max(1, len(losses_window))
                    losses_window.clear()
                    elapsed = time.time() - t_start
                    pbar.set_postfix(loss=f"{avg:.4f}", sec=f"{elapsed:.0f}")

                if step % args.save_every == 0 or step == max_steps:
                    out = out_dir / f"temporal_lllite_step{step:06d}.safetensors"
                    from safetensors.torch import save_file
                    sd = {k: v.detach().cpu().to(torch.float32) for k, v in controller.state_dict().items()}
                    md = {
                        "kind": "temporal_lllite",
                        "cond_emb_dim": str(args.cond_emb_dim),
                        "mlp_dim": str(args.mlp_dim),
                        "step": str(step),
                        "model": args.model,
                    }
                    save_file(sd, str(out), md)
                    pbar.write(f"  saved {out}")

                if step >= max_steps:
                    break
        if step >= max_steps:
            break

    pbar.close()
    print(f"done: trained for {step} steps in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
