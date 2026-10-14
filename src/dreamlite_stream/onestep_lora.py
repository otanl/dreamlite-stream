"""One-step LoRA: low-rank weight deltas that make ONE denoising step
reproduce the UNet's own multi-step output.

Why this is cheap here. On the lean path (see `lean.py`) every frame uses
the same init-noise tensor and the prompt embedding is fixed, so the K-step
output is a deterministic function of the input frame: for each training
frame there is exactly one target. Distilling K steps into one is then
plain paired regression, with none of the mode-averaging that blurs
one-step students trained on random noise. `scripts/train_onestep_lora.py`
trains the deltas in minutes on one GPU.

At run time the deltas are MERGED into the UNet weights (`merge`). There is
no adapter module left in the forward pass, so the merged UNet costs exactly
what the base UNet costs, and torch.compile or an ONNX export see an
ordinary UNet. `MergedLoRA.restore()` puts the original weights back.

A LoRA is trained for one resolution, one init-noise tensor and the prompts
in its metadata; `compatibility_notes` reports mismatches against how a
worker is about to be run. The noise tensor is stored in the file
(`init_noise`) and used as the worker's shared noise, because a seed alone
does not pin it down: seeded draws differ between torch builds.

File format: safetensors with `<module path>.lora_A` (rank, in) and
`<module path>.lora_B` (out, rank) per adapted layer, `init_noise`
(1, C, h, w), and string metadata (JSON-encoded values) describing how it
was trained.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Layer groups a LoRA can adapt, by module-path pattern.
#   attn  attention projections (to_q / to_k / to_v / to_out.0), 25% of the UNet
#   ff    transformer feed-forward Linears, 51% of the UNet
#   pw    pointwise (1x1) convolutions of the ResNet blocks, 11% of the UNet
NOISE_KEY = "init_noise"   # the shared init-noise tensor a LoRA was fitted to

TARGET_PATTERNS = {
    "attn": re.compile(r"\.attn[12]\.to_(q|k|v)$|\.attn[12]\.to_out\.0$"),
    "ff": re.compile(r"\.ff\.net\.(0\.proj|2)$"),
    "pw": re.compile(r"\.resnets\.\d+\.conv\d+\.pointwise$"),
}


def _is_target(name: str, module: nn.Module, targets: Iterable[str]) -> bool:
    for t in targets:
        if t not in TARGET_PATTERNS:
            raise ValueError(f"unknown LoRA target group {t!r}; choose from {sorted(TARGET_PATTERNS)}")
        if not TARGET_PATTERNS[t].search(name):
            continue
        if isinstance(module, nn.Linear):
            return True
        if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and module.groups == 1:
            return True
    return False


class LoRAWrapper(nn.Module):
    """A Linear or 1x1 Conv2d plus a trainable low-rank residual (training only).

    `B` starts at zero, so a fresh wrapper is the identity change. The
    residual runs in the parameters' dtype (fp32) regardless of the host's.
    """

    def __init__(self, host: nn.Module, rank: int, alpha: float):
        super().__init__()
        self.host = host
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = 1.0
        for p in self.host.parameters():
            p.requires_grad = False
        out_dim, in_dim = host.weight.shape[0], host.weight.shape[1]
        self.A = nn.Parameter(torch.zeros(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.host(x)
        # Under autocast the two low-rank ops run in the autocast dtype by
        # themselves; an explicit fp32 copy of the activation would only cost
        # memory (the FF inputs at the top resolution are large).
        xa = x if torch.is_autocast_enabled() else x.to(self.A.dtype)
        if isinstance(self.host, nn.Linear):
            r = F.linear(F.linear(xa, self.A), self.B)
        else:
            r = F.conv2d(F.conv2d(xa, self.A[:, :, None, None]), self.B[:, :, None, None])
        return h + (self.scale * self.alpha / self.rank) * r.to(h.dtype)


def attach(unet: nn.Module, rank: int = 16, alpha: Optional[float] = None,
           targets: Iterable[str] = ("attn",)) -> Dict[str, LoRAWrapper]:
    """Wrap every target layer of `unet` in a LoRAWrapper. Returns {module path: wrapper}."""
    alpha = float(rank) if alpha is None else float(alpha)
    targets = tuple(targets)
    found = [(n, m) for n, m in unet.named_modules() if _is_target(n, m, targets)]
    wrappers: Dict[str, LoRAWrapper] = {}
    for name, module in found:
        parent_name, attr = name.rsplit(".", 1)
        wrapper = LoRAWrapper(module, rank, alpha).to(module.weight.device)
        setattr(unet.get_submodule(parent_name), attr, wrapper)
        wrappers[name] = wrapper
    if not wrappers:
        raise ValueError(f"no layers matched LoRA targets {targets}")
    return wrappers


def detach(unet: nn.Module, wrappers: Dict[str, LoRAWrapper]) -> None:
    """Undo `attach`: put the original layers back."""
    for name, wrapper in wrappers.items():
        parent_name, attr = name.rsplit(".", 1)
        setattr(unet.get_submodule(parent_name), attr, wrapper.host)


def state_from_wrappers(wrappers: Dict[str, LoRAWrapper]) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, w in wrappers.items():
        state[f"{name}.lora_A"] = w.A.detach().float().cpu().contiguous()
        state[f"{name}.lora_B"] = w.B.detach().float().cpu().contiguous()
    return state


def save(path, state: Dict[str, torch.Tensor], meta: dict) -> None:
    from safetensors.torch import save_file
    save_file(state, str(path), metadata={k: json.dumps(v) for k, v in meta.items()})


def load(path) -> Tuple[Dict[str, torch.Tensor], dict]:
    from safetensors import safe_open
    state: Dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        raw = f.metadata() or {}
        for k in f.keys():
            state[k] = f.get_tensor(k)
    meta = {}
    for k, v in raw.items():
        try:
            meta[k] = json.loads(v)
        except (TypeError, ValueError):
            meta[k] = v
    return state, meta


def init_noise(state: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """The init-noise tensor stored with a LoRA, or None for a file written before it was stored."""
    return state.get(NOISE_KEY)


def load_init_noise(path) -> torch.Tensor:
    """Read just the `init_noise` tensor of a safetensors file (a LoRA, or an engine's .noise file)."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        if NOISE_KEY not in f.keys():
            raise ValueError(f"{path} holds no '{NOISE_KEY}' tensor")
        return f.get_tensor(NOISE_KEY)


def layer_names(state: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(k[: -len(".lora_A")] for k in state if k.endswith(".lora_A"))


def delta_weight(state: Dict[str, torch.Tensor], name: str, alpha_over_rank: float,
                 like: torch.Tensor) -> torch.Tensor:
    """The fp32 weight delta of one layer, shaped like the layer's weight."""
    a = state[f"{name}.lora_A"].to(device=like.device, dtype=torch.float32)
    b = state[f"{name}.lora_B"].to(device=like.device, dtype=torch.float32)
    return ((b @ a) * alpha_over_rank).reshape(like.shape)


@dataclass
class MergedLoRA:
    """Handle returned by `merge`. `restore()` puts the original weights back exactly."""

    meta: dict
    n_layers: int
    _unet: Optional[nn.Module] = field(default=None, repr=False)
    _backup: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    def restore(self) -> None:
        if self._unet is None:
            return
        with torch.no_grad():
            for name, w in self._backup.items():
                self._unet.get_submodule(name).weight.copy_(w)
        self._backup = {}
        self._unet = None


def merge(unet: nn.Module, state: Dict[str, torch.Tensor], meta: Optional[dict] = None,
          scale: float = 1.0) -> MergedLoRA:
    """Add the LoRA deltas to `unet`'s weights in place.

    Must run before torch.compile captures the UNet. The sum is formed in
    fp32 and rounded once to the weight's own dtype. Works on a UNet whose
    Linear forwards have been patched (e.g. by LLLite): only `.weight` is
    touched, module identity is kept.
    """
    meta = dict(meta or {})
    unet = getattr(unet, "_orig_mod", unet)  # tolerate an already-compiled wrapper
    rank = meta.get("rank")
    alpha = meta.get("alpha")
    backup: Dict[str, torch.Tensor] = {}
    names = layer_names(state)
    if not names:
        raise ValueError("LoRA file holds no '<layer>.lora_A' entries")
    with torch.no_grad():
        for name in names:
            w = unet.get_submodule(name).weight
            r = state[f"{name}.lora_A"].shape[0]
            a_over_r = (float(alpha) / float(rank)) if (alpha and rank) else 1.0
            if rank and r != rank:
                raise ValueError(f"{name}: rank {r} in file, {rank} in metadata")
            delta = delta_weight(state, name, a_over_r * scale, w)
            backup[name] = w.detach().clone()
            w.copy_((w.float() + delta).to(w.dtype))
    return MergedLoRA(meta=meta, n_layers=len(names), _unet=unet, _backup=backup)


def compatibility_notes(meta: dict, *, size: int, seed: Optional[int], steps: int,
                        shared_noise: bool, prompt: Optional[str] = None,
                        has_noise: bool = False) -> List[str]:
    """Human-readable mismatches between how a LoRA was trained and how it will be run.

    `has_noise`: the file carries its init-noise tensor and the worker will
    use it, so the seed no longer matters.
    """
    notes: List[str] = []
    if meta.get("size") not in (None, size):
        notes.append(f"trained at {meta['size']}x{meta['size']}, running at {size}x{size}")
    if not has_noise and meta.get("seed") is not None:
        if seed != meta["seed"]:
            notes.append(f"trained for noise seed {meta['seed']}, running with seed {seed}")
        notes.append("the file stores no init-noise tensor; it fits only where this torch build draws, for "
                     "that seed, the tensor it was trained on (seeded draws differ between builds)")
    if not shared_noise:
        notes.append("trained for one shared init-noise tensor; pass --shared_noise (or --lean)")
    if steps != 1:
        notes.append(f"trained for 1 denoising step, running {steps}")
    prompts = meta.get("prompts")
    if prompt is not None and prompts and prompt not in prompts:
        notes.append("the prompt is not one of the prompts it was trained on")
    return notes
