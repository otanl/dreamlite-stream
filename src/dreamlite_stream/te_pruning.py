"""Visual-token pruning hooks for the Qwen3-VL text encoder (TE).

Two methods are scaffolded here:

- :class:`IDPrunerHook` — implements the IDPruner MMR-style selection
  (Tan et al. 2026, arXiv:2602.13315): pick a fixed budget of visual
  tokens that maximises ``lambda * importance - (1-lambda) * max_sim``.
  Attention-map-free, FlashAttention-compatible.
- :class:`HAWKHook` — placeholder for the HAWK head-importance method
  (Zhu et al. 2026, arXiv:2604.07812). Code release pending upstream
  (``github.com/peppery77/HAWK`` is currently README-only).

Both hooks operate at the boundary between the Qwen3-VL projector and
the LLM stack: the visual-token slice of ``inputs_embeds`` (positions
where ``input_ids == image_token_id``) is replaced by a pruned slice
padded to a fixed budget ``K``. Fixed-K padding is mandatory under
``compile_mode="reduce-overhead"`` — variable-length output would force
CUDA-graph recapture every refresh.

See ``notes/te_pruning_integration.md`` for the integration plan,
expected speedup contour (~250 ms → 110-130 ms uncached at 75%
pruning on 3090 Ti), and the Day 1-5 validation matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class IDPrunerConfig:
    """IDPruner configuration.

    Attributes:
        budget_k: target visual-token count after pruning. Choose to be a
            constant under ``reduce-overhead`` so CUDA-graph capture stays
            stable. Typical values for Qwen3-VL @ 256x256 input (~256
            visual tokens): 192 (25% prune), 128 (50%), 64 (75%), 32 (87%).
        lambda_balance: MMR balance between importance and diversity.
            1.0 = pure importance (collapses to top-K), 0.0 = pure
            diversity (greedy farthest-point sampling). Paper default 0.7.
        importance_metric: how to score per-token importance. ``"l2"``
            uses the projected visual embedding L2 norm (cheap, no extra
            matmul); ``"attn_diag"`` uses a forward through one attention
            layer to score (more expensive, may need eager attn).
        pad_with: how to fill positions beyond ``budget_k``. ``"zero"``
            zeros the embeddings and zeros the attention mask; ``"learned"``
            would use a trained pad embedding (not implemented).
    """

    budget_k: int = 128
    lambda_balance: float = 0.7
    importance_metric: str = "l2"
    pad_with: str = "zero"

    def __post_init__(self) -> None:
        if self.budget_k <= 0:
            raise ValueError(f"budget_k must be positive; got {self.budget_k}")
        if not (0.0 <= self.lambda_balance <= 1.0):
            raise ValueError(f"lambda_balance must be in [0, 1]; got {self.lambda_balance}")
        if self.importance_metric not in {"l2", "attn_diag"}:
            raise ValueError(f"unknown importance_metric: {self.importance_metric}")
        if self.pad_with not in {"zero", "learned"}:
            raise ValueError(f"unknown pad_with: {self.pad_with}")
        if self.pad_with == "learned":
            raise NotImplementedError("learned pad embedding not implemented yet")


@dataclass
class HAWKConfig:
    """HAWK configuration (placeholder).

    HAWK requires an offline head-importance calibration step that produces
    a per-head weight tensor; pruning at inference is text-guided attention
    weighted by those head importances.
    """

    budget_k: int = 128
    head_importance_path: Optional[str] = None  # offline calibration output


# ---------------------------------------------------------------------------
# Visual-token identification
# ---------------------------------------------------------------------------


def find_visual_token_span(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Per-row boolean mask of positions that are visual tokens.

    Qwen3-VL tokenizer replaces image placeholders with the per-patch
    ``image_token_id`` repeated for the grid_thw count. After
    ``get_image_features`` runs, the corresponding ``inputs_embeds`` slice
    holds the projector output; we identify it by matching the input_ids.

    Shape: ``input_ids`` (B, L) → returns (B, L) boolean.
    """
    return input_ids == image_token_id


# ---------------------------------------------------------------------------
# IDPruner scoring
# ---------------------------------------------------------------------------


def _l2_importance(visual_tokens: torch.Tensor) -> torch.Tensor:
    """Per-token L2 norm in the projected visual-embedding space.

    Shape: (B, V, D) → (B, V). Min-max normalised per row.
    """
    norms = visual_tokens.float().norm(dim=-1)  # (B, V)
    lo = norms.amin(dim=-1, keepdim=True)
    hi = norms.amax(dim=-1, keepdim=True)
    return (norms - lo) / (hi - lo + 1e-6)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity (B, V_a, D) × (B, V_b, D) → (B, V_a, V_b)."""
    a_n = F.normalize(a.float(), dim=-1)
    b_n = F.normalize(b.float(), dim=-1)
    return torch.bmm(a_n, b_n.transpose(1, 2))


def idpruner_select(
    visual_tokens: torch.Tensor,
    cfg: IDPrunerConfig,
) -> torch.Tensor:
    """MMR-style selection: pick ``cfg.budget_k`` indices per batch row.

    Shape: ``visual_tokens`` (B, V, D) → returns (B, K) long indices into
    the V dimension.

    Algorithm (per batch element):
        selected = []
        for k in range(K):
            score[v] = lambda * Imp(v)
                     - (1-lambda) * max_{s in selected} sim(v, s)
            selected.append(argmax_{v not in selected} score[v])

    Greedy O(K*V) per row. Vectorised over B.
    """
    if cfg.importance_metric == "attn_diag":
        raise NotImplementedError(
            "attn_diag importance not implemented; use 'l2' for the dry-run."
        )

    B, V, _ = visual_tokens.shape
    K = min(cfg.budget_k, V)
    importance = _l2_importance(visual_tokens)  # (B, V)
    sim = _cosine_sim(visual_tokens, visual_tokens)  # (B, V, V)
    sim.diagonal(dim1=1, dim2=2).fill_(0.0)  # ignore self-similarity

    device = visual_tokens.device
    selected = torch.full((B, K), -1, dtype=torch.long, device=device)
    available = torch.ones((B, V), dtype=torch.bool, device=device)
    max_sim_to_selected = torch.zeros((B, V), device=device)

    lam = cfg.lambda_balance
    for k in range(K):
        score = lam * importance - (1.0 - lam) * max_sim_to_selected
        score = score.masked_fill(~available, float("-inf"))
        idx = score.argmax(dim=-1)  # (B,)
        selected[:, k] = idx
        batch_arange = torch.arange(B, device=device)
        available[batch_arange, idx] = False
        new_col = sim[batch_arange, :, idx]  # (B, V)
        max_sim_to_selected = torch.maximum(max_sim_to_selected, new_col)

    return selected


# ---------------------------------------------------------------------------
# Fixed-K padding
# ---------------------------------------------------------------------------


def prune_and_pad(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    visual_mask: torch.Tensor,
    selected: torch.Tensor,
    budget_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the visual-token slice with the K selected tokens (zero-padded).

    Shape:
        inputs_embeds  : (B, L, D)
        attention_mask : (B, L)
        visual_mask    : (B, L) bool
        selected       : (B, K) long indices into the V dimension

    Returns updated (inputs_embeds, attention_mask) of the SAME L: the
    selected K visual embeddings replace the first K visual-mask positions
    per row, the remaining (V-K) visual-mask positions are zeroed in both
    inputs_embeds and attention_mask.

    Keeping L constant is the compile-graph stability requirement; with
    fixed K and fixed L the CUDA-graph capture sees a constant shape.
    """
    B, L, D = inputs_embeds.shape
    out_embeds = inputs_embeds.clone()
    out_mask = attention_mask.clone()

    for b in range(B):
        visual_positions = visual_mask[b].nonzero(as_tuple=False).squeeze(-1)  # (V,)
        V = visual_positions.numel()
        K = min(budget_k, V)
        keep_positions = visual_positions[: K]
        drop_positions = visual_positions[K:]

        sel = selected[b, :K]
        chosen_embeds = inputs_embeds[b, visual_positions][sel]  # (K, D)
        out_embeds[b, keep_positions] = chosen_embeds
        if drop_positions.numel() > 0:
            out_embeds[b, drop_positions] = 0.0
            out_mask[b, drop_positions] = 0

    return out_embeds, out_mask


# ---------------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------------


def _find_language_model(qwen3vl_model: torch.nn.Module) -> torch.nn.Module:
    """Locate the text decoder stack on a Qwen-VL-family model.

    The hook point is the *language model* entry: by then the vision
    tower's features have been merged into ``inputs_embeds`` (scattered
    over the image-token positions) but the attention mask is still 2D
    — exactly the contract :func:`prune_and_pad` assumes. Hooking a
    decoder *layer* instead would face a 4D causal mask and per-layer
    rope state, which is why we attach one level up.
    """
    for path in ("model.language_model", "language_model", "model"):
        m = qwen3vl_model
        ok = True
        for attr in path.split("."):
            m = getattr(m, attr, None)
            if m is None:
                ok = False
                break
        if ok and m is not None and hasattr(m, "layers"):
            return m
    raise AttributeError(
        "could not locate the language-model stack on "
        f"{type(qwen3vl_model).__name__}; probed model.language_model, "
        "language_model, model"
    )


class IDPrunerHook:
    """Two-hook IDPruner installation for Qwen-VL-family text encoders.

    Usage::

        hook = IDPrunerHook(IDPrunerConfig(budget_k=64))
        hook.attach(text_encoder, image_token_id=image_id)
        # ... TE forwards now see a pruned visual slice ...
        hook.detach()

    Hook (a) on the top-level model captures ``input_ids`` (the decoder
    stack receives embeds, not ids). Hook (b) on the language-model entry
    rewrites ``inputs_embeds`` / ``attention_mask`` via
    :func:`idpruner_select` + :func:`prune_and_pad`. Sequence length L is
    left unchanged (zero-padded), so downstream position ids, rope state,
    and any captured CUDA graph keep their shapes.

    Contrast with the Mobile-O integration
    (``scripts/mobile_o/test_idpruner_mobileo.py``): there the projected
    visual tokens exist as a standalone tensor *before* the splice, so a
    plain wrapper suffices; here the merge happens inside the model
    forward, hence the hooks. Same component, different attach point —
    the TE-family-axis asymmetry discussed in the paper.
    """

    def __init__(self, cfg: IDPrunerConfig) -> None:
        self.cfg = cfg
        self.last_input_ids: Optional[torch.Tensor] = None
        self.last_stats: Optional[dict] = None
        self._handle_capture = None
        self._handle_prune = None

    def attach(self, qwen3vl_model: torch.nn.Module, image_token_id: int):
        def _capture(module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is not None:
                self.last_input_ids = ids
            return None

        def _prune(module, args, kwargs):
            embeds = kwargs.get("inputs_embeds")
            if embeds is None or self.last_input_ids is None:
                return None
            ids = self.last_input_ids
            if ids.shape[0] != embeds.shape[0] or ids.shape[1] != embeds.shape[1]:
                return None  # shape mismatch (e.g. generation step) — skip
            visual_mask = find_visual_token_span(ids, image_token_id)
            if not visual_mask.any():
                return None
            counts = visual_mask.sum(dim=-1)
            v = int(counts[0].item())
            if not bool((counts == v).all()):
                raise NotImplementedError(
                    "per-row visual-token counts differ; the vectorised "
                    "selection assumes a uniform V (true under the fixed "
                    "processor padding this pipeline uses)"
                )
            if v <= self.cfg.budget_k:
                return None  # nothing to prune

            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None or attention_mask.dim() != 2:
                raise NotImplementedError(
                    "IDPrunerHook expects a 2D attention_mask at the "
                    "language-model entry; got "
                    f"{None if attention_mask is None else attention_mask.shape}"
                )

            B, L, D = embeds.shape
            visual_tokens = embeds[visual_mask].view(B, v, D)
            selected = idpruner_select(visual_tokens, self.cfg)
            selected, _ = selected.sort(dim=-1)  # keep spatial order
            new_embeds, new_mask = prune_and_pad(
                embeds, attention_mask, visual_mask, selected,
                self.cfg.budget_k)
            self.last_stats = {"V": v, "K": int(selected.shape[1]), "L": L}
            kwargs["inputs_embeds"] = new_embeds
            kwargs["attention_mask"] = new_mask
            return (args, kwargs)

        self._handle_capture = qwen3vl_model.register_forward_pre_hook(
            _capture, with_kwargs=True)
        lm = _find_language_model(qwen3vl_model)
        self._handle_prune = lm.register_forward_pre_hook(
            _prune, with_kwargs=True)
        return self

    def detach(self) -> None:
        for h in (self._handle_capture, self._handle_prune):
            if h is not None:
                h.remove()
        self._handle_capture = None
        self._handle_prune = None


class HAWKHook:
    """Placeholder. HAWK code at ``github.com/peppery77/HAWK`` is currently
    README-only; install once code is released.
    """

    def __init__(self, cfg: HAWKConfig) -> None:
        self.cfg = cfg

    def attach(self, qwen3vl_model: torch.nn.Module, image_token_id: int):
        raise NotImplementedError(
            "HAWK code not yet released (as of 2026-06-05). "
            "Track github.com/peppery77/HAWK for the public release."
        )
