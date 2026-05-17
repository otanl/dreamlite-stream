# Cross-Frame Attention Adapter — Design Notes

## Motivation

LLLite injects a per-attention `delta` based on a CNN-encoded conditioning
image (here: the warped previous output). This is *content-conditioning*:
the adapter learns "given this picture, modulate the attention output."

A **cross-frame attention adapter** instead lets the current frame's
attention *directly attend* to features from the previous frame. This is
how AnimateDiff / temporal-attention layers work — but at the level of an
adapter rather than a retrained model.

## Core mechanism

For each self-attention block:
```
Q_t  = host.to_q(x_t)        # current frame
K_t  = host.to_k(x_t)
V_t  = host.to_v(x_t)
K_prev = host.to_k(x_{t-1})  # cached from previous frame
V_prev = host.to_v(x_{t-1})

# Standard attention on current frame:
A_self = softmax(Q_t @ K_t.T / sqrt(d)) @ V_t

# NEW: cross-frame attention to previous frame:
A_cross = softmax(Q_t @ K_prev.T / sqrt(d)) @ V_prev

# Adapter mixes them:
delta = adapter(A_self, A_cross)  # small MLP or learned gate
output = host.to_out(A_self + delta)
```

## Why this might be better than LLLite

- LLLite cond is at adapter granularity (one CNN per hook). Cross-frame
  attention lets information flow at full attention bandwidth.
- No need to re-encode a cond image every frame — just cache previous
  frame's K, V tensors (cheap, on-device memory).
- Naturally compositional with batched inference: in batched call,
  cross-frame attention is just an extra small attention op.

## Implementation challenges

1. **K/V caching**: need a global cache per attention block, updated each
   frame. Not compile-friendly without explicit buffer management.
2. **Variable cond shape**: K_prev shape changes if image size changes.
   Lock to fixed size (matching our 512² setup).
3. **Training data**: same as LLLite — frame pairs + teacher targets.
4. **Memory**: caching K/V at every attention block × every frame =
   ~108 cached tensors. Each is ~1 MB at 512². Total ~110 MB. OK.

## Time estimate

- Implement K/V cache + cross-attention compute: 2-3 days
- Wire into LLLite-style training loop: 1 day
- Train: 1 day
- Eval: 1 day
- **Total: ~1 week**

## Comparison to LLLite + temporal cond

| Aspect | LLLite + warped prev | Cross-frame attn |
|---|---|---|
| Cond representation | CNN-encoded image | Direct K/V tensors |
| Cond preparation | conditioning1 CNN per hook | Cache, no recompute |
| Compile-friendliness | Achieved with buffer trick | Harder (dynamic K/V) |
| Information bandwidth | One vector per spatial pos | Full attention to prev features |
| Training data | (input, warp(prev_out)) | (input, prev_K, prev_V) |
| Memory overhead | ~5 MB per hook × 108 | ~1 MB per hook × 108 |
| Bottleneck | conditioning1 CNN cost | Doubled attention cost |

## Open questions

- Does the model benefit from looking *back* multiple frames?
  Could cache last K frames and attend to all (windowed cross-frame attn).
- Can we share the cross-frame attention budget across the batch?
  (No — each frame in batch should attend to its own predecessor.)
