"""Compile only the vision-encoder portion of Qwen3-VL.

Follows the vLLM pattern (see notes/te_pruning_integration.md and the
DEV.to write-up "Compiling the Vision Encoder", 2026): compile the ViT
submodule of a multimodal LM independently of the language tower so the
language-tower's sliding-attention / KV-cache code path doesn't trip on
``torch.compile`` quirks. Reported ~3-4% end-to-end throughput on Hopper
when the encoder is only ~13% of inference; smaller relative gain on
Ampere/Ada but still "free" once integrated.

Composes orthogonally with :mod:`dreamlite_stream.te_pruning` (which
attaches a forward-pre-hook on the LLM's first decoder layer) and with
:mod:`dreamlite_stream.sage_attn` (which can be the underlying SDPA
backend inside the compiled vision tower).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


_KNOWN_VIT_ATTRS = (
    "visual",          # Qwen3-VL: ``model.visual`` is the ViT tower
    "vision_tower",    # Llava-style nomenclature, kept as fallback
    "vision_model",    # SigLIP-style nomenclature
)


@dataclass
class CompileConfig:
    """Compile-time options for the ViT.

    Attributes:
        mode: torch.compile mode. ``"default"`` for safety, ``"reduce-
            overhead"`` for CUDA-graph capture. SageAttention issues #74 /
            #162 mean reduce-overhead may need the custom-op wrapper from
            :mod:`dreamlite_stream.sage_attn` if SageAttention is also in
            play; otherwise reduce-overhead is fine here because the ViT
            sees fixed-shape pixel_values from the processor.
        dynamic: ``False`` matches our fixed 256x256 ViT input from the
            processor; flip to ``True`` only if input resolution varies.
        fullgraph: ``False`` is safer; the ViT has a handful of Python-
            side branches (e.g. classifier head off in encoder-only mode).
    """

    mode: str = "reduce-overhead"
    dynamic: bool = False
    fullgraph: bool = False


def find_vision_tower(qwen3vl_model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Locate the ViT submodule on a Qwen3-VL instance.

    Probes the known attribute names; returns ``None`` if none match
    (the caller should then inspect the model and add a new attr to
    ``_KNOWN_VIT_ATTRS``).
    """
    for attr in _KNOWN_VIT_ATTRS:
        m = getattr(qwen3vl_model, attr, None)
        if m is None:
            inner = getattr(qwen3vl_model, "model", None)
            if inner is not None:
                m = getattr(inner, attr, None)
        if isinstance(m, torch.nn.Module):
            return m
    return None


def compile_vision_tower(
    qwen3vl_model: torch.nn.Module,
    cfg: CompileConfig = CompileConfig(),
) -> bool:
    """Compile the ViT in place. Returns True on success, False if no ViT
    tower was found (e.g. the model uses a non-standard attribute name).

    On success the caller can verify with::

        from dreamlite_stream.te_vit_compile import compile_vision_tower
        ok = compile_vision_tower(pipeline.text_encoder)
        # first forward triggers compilation; subsequent forwards run on
        # the captured CUDA graph

    Raises any exception from torch.compile itself (does not swallow).
    """
    tower = find_vision_tower(qwen3vl_model)
    if tower is None:
        return False

    parent_attr = next(
        attr for attr in _KNOWN_VIT_ATTRS if getattr(qwen3vl_model, attr, None) is tower
    ) if any(getattr(qwen3vl_model, a, None) is tower for a in _KNOWN_VIT_ATTRS) else None

    compiled = torch.compile(
        tower, mode=cfg.mode, dynamic=cfg.dynamic, fullgraph=cfg.fullgraph,
    )

    if parent_attr is not None:
        setattr(qwen3vl_model, parent_attr, compiled)
    else:
        inner = getattr(qwen3vl_model, "model")
        for attr in _KNOWN_VIT_ATTRS:
            if getattr(inner, attr, None) is tower:
                setattr(inner, attr, compiled)
                break

    return True
