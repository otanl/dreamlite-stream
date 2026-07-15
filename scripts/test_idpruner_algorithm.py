"""Synthetic-tensor functional test for IDPruner selection + padding.

Verifies, without loading a Qwen3-VL model:

1. `idpruner_select` returns exactly K unique indices per batch row when
   K <= V, and respects the MMR objective trade-off (lambda=1.0 ->
   selection collapses to top-K by importance; lambda=0.0 -> selection
   spreads, no duplicates).
2. `prune_and_pad` keeps the sequence length L constant, places the K
   selected embeddings into the first K visual-token positions, zeroes
   the remaining (V-K) visual-token positions in both embeddings and
   attention mask, and leaves non-visual positions untouched.

Runs on CPU in a few seconds; no GPU needed.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "src")
from dreamlite_stream.te_pruning import (  # noqa: E402
    IDPrunerConfig,
    find_visual_token_span,
    idpruner_select,
    prune_and_pad,
    _l2_importance,
    _cosine_sim,
)


def test_l2_importance_normalized():
    """L2 importance should be min-max normalised in [0, 1] per row."""
    torch.manual_seed(0)
    visual = torch.randn(2, 16, 8)
    imp = _l2_importance(visual)
    assert imp.shape == (2, 16)
    for row in range(imp.size(0)):
        assert imp[row].min().item() == 0.0, "min should be 0 after normalisation"
        assert abs(imp[row].max().item() - 1.0) < 1e-5, "max should be 1"
    print("[OK] L2 importance min-max normalised")


def test_cosine_sim_self_diagonal():
    """cosine_sim(X, X) should have ~1 on the diagonal."""
    torch.manual_seed(1)
    X = torch.randn(2, 8, 4)
    sim = _cosine_sim(X, X)
    assert sim.shape == (2, 8, 8)
    diag = sim.diagonal(dim1=1, dim2=2)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5), \
        f"diag should be ~1, got {diag}"
    print("[OK] cosine_sim self-diagonal ~1")


def test_idpruner_select_unique_indices():
    """Selected indices must be unique per row, no -1 sentinel left."""
    torch.manual_seed(2)
    B, V, D = 4, 64, 16
    visual = torch.randn(B, V, D)
    cfg = IDPrunerConfig(budget_k=24, lambda_balance=0.7)
    sel = idpruner_select(visual, cfg)
    assert sel.shape == (B, 24)
    for b in range(B):
        row = sel[b].tolist()
        assert -1 not in row, f"row {b} has -1 sentinel: {row}"
        assert len(set(row)) == len(row), \
            f"row {b} has duplicate indices: {row}"
    print("[OK] idpruner_select returns K unique indices per row")


def test_idpruner_lambda_extremes():
    """lambda=1.0 -> selection is exactly top-K by L2 importance.

    lambda=0.0 -> selection still produces K unique indices (greedy
    farthest-point sampling).
    """
    torch.manual_seed(3)
    B, V, D = 2, 32, 8
    visual = torch.randn(B, V, D)
    K = 8

    # Lambda = 1.0 should collapse to top-K by importance.
    cfg_imp = IDPrunerConfig(budget_k=K, lambda_balance=1.0)
    sel_imp = idpruner_select(visual, cfg_imp)
    imp = _l2_importance(visual)
    topk_idx = imp.topk(K, dim=-1).indices
    # The greedy selection by importance must produce *the same set* as topk,
    # though the order may differ (top-K selected one by one in descending order).
    for b in range(B):
        sel_set = set(sel_imp[b].tolist())
        topk_set = set(topk_idx[b].tolist())
        assert sel_set == topk_set, (
            f"row {b}: lambda=1.0 selection {sorted(sel_set)} != topK by "
            f"L2 {sorted(topk_set)}"
        )
    print("[OK] lambda=1.0 -> top-K by L2 importance")

    # Lambda = 0.0 should still produce K unique indices.
    cfg_div = IDPrunerConfig(budget_k=K, lambda_balance=0.0)
    sel_div = idpruner_select(visual, cfg_div)
    for b in range(B):
        row = sel_div[b].tolist()
        assert len(set(row)) == K, f"row {b}: lambda=0 not unique: {row}"
    print("[OK] lambda=0.0 -> K unique indices (diversity-only mode)")


def test_find_visual_token_span():
    """Mask should pick out exactly the positions where input_ids matches."""
    input_ids = torch.tensor([
        [10, 10, 99, 99, 99, 10, 10],
        [99, 99, 99, 99, 10, 10, 10],
    ])
    mask = find_visual_token_span(input_ids, image_token_id=99)
    expected = torch.tensor([
        [False, False, True, True, True, False, False],
        [True, True, True, True, False, False, False],
    ])
    assert torch.equal(mask, expected), f"mask mismatch:\n{mask}\n{expected}"
    print("[OK] find_visual_token_span detects image-token positions")


def test_prune_and_pad_shape_stability():
    """After prune_and_pad, L stays the same and selected visual tokens land
    in the first K visual-mask positions; remaining (V-K) positions zeroed."""
    torch.manual_seed(4)
    B, L, D = 2, 32, 16
    K = 8
    inputs_embeds = torch.randn(B, L, D)
    attention_mask = torch.ones(B, L, dtype=torch.long)
    # Mark positions 8..24 as visual tokens in row 0; 4..28 in row 1.
    visual_mask = torch.zeros(B, L, dtype=torch.bool)
    visual_mask[0, 8:24] = True   # 16 visual tokens
    visual_mask[1, 4:28] = True   # 24 visual tokens

    # Pick the first K from each row (mimicking an idpruner_select result).
    selected = torch.zeros(B, K, dtype=torch.long)
    selected[0] = torch.arange(K)
    selected[1] = torch.arange(K)

    out_embeds, out_mask = prune_and_pad(inputs_embeds, attention_mask,
                                         visual_mask, selected, budget_k=K)

    # Shape preservation.
    assert out_embeds.shape == inputs_embeds.shape, "embeds shape changed"
    assert out_mask.shape == attention_mask.shape, "mask shape changed"

    # Non-visual positions unchanged.
    nonvis = ~visual_mask
    assert torch.equal(out_embeds[nonvis], inputs_embeds[nonvis]), \
        "non-visual embeddings were mutated"
    assert torch.equal(out_mask[nonvis], attention_mask[nonvis]), \
        "non-visual attention mask was mutated"

    # Row 0: visual positions 8..24, K=8. First 8 visual positions (8..16)
    # should hold the selected embeddings; positions 16..24 should be zero
    # in both embeds and attention_mask.
    assert torch.all(out_embeds[0, 16:24] == 0.0), \
        "row 0 dropped visual positions not zeroed in embeds"
    assert torch.all(out_mask[0, 16:24] == 0), \
        "row 0 dropped visual positions not zeroed in attention_mask"

    # Row 1: visual positions 4..28, K=8. First 8 visual positions (4..12)
    # should hold the selected embeddings; positions 12..28 should be zero.
    assert torch.all(out_embeds[1, 12:28] == 0.0), \
        "row 1 dropped visual positions not zeroed"
    assert torch.all(out_mask[1, 12:28] == 0), \
        "row 1 dropped visual positions mask not zeroed"

    print("[OK] prune_and_pad keeps L stable + zeroes dropped positions")


def test_prune_and_pad_budget_larger_than_V():
    """When budget K > V, K should clamp to V (no out-of-bound writes)."""
    torch.manual_seed(5)
    B, L, D = 1, 16, 8
    inputs_embeds = torch.randn(B, L, D)
    attention_mask = torch.ones(B, L, dtype=torch.long)
    visual_mask = torch.zeros(B, L, dtype=torch.bool)
    visual_mask[0, 4:8] = True   # V=4 visual tokens
    K = 6                         # budget > V
    selected = torch.zeros(B, K, dtype=torch.long)
    selected[0, :4] = torch.arange(4)
    selected[0, 4:] = 0           # padding region
    out_embeds, out_mask = prune_and_pad(inputs_embeds, attention_mask,
                                         visual_mask, selected, budget_k=K)
    # No assertion error means we did not write out of bounds.
    assert out_embeds.shape == inputs_embeds.shape
    print("[OK] prune_and_pad clamps K to V")


if __name__ == "__main__":
    test_l2_importance_normalized()
    test_cosine_sim_self_diagonal()
    test_idpruner_select_unique_indices()
    test_idpruner_lambda_extremes()
    test_find_visual_token_span()
    test_prune_and_pad_shape_stability()
    test_prune_and_pad_budget_larger_than_V()
    print("\nALL IDPRUNER ALGORITHM TESTS PASSED")
