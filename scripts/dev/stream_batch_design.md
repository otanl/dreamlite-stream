# Stream Batch — Design Notes (DreamLite-mobile adaptation)

## Motivation

StreamDiffusion's headline 91~FPS @ 512² on a 4090 comes from
**Stream Batch (SB)**: across the K-step denoise trajectory, run K
frames concurrently — frame at step t=0, frame_{-1} at step t=1, ...
frame_{-K+1} at step t=K-1 — sharing the GPU compute footprint. After
K steps, every frame has progressed exactly one step, and a fully
denoised frame_{-K+1} is emitted while a fresh frame_0 enters the
pipeline.

For our setup (DreamLite-mobile, K=1 free-1-step), naive SB does
nothing because K=1. **But our champion uses K=1 only for inference
speed; the trajectory is logically 4-step.** We can re-introduce K=2
or K=3 inference and then stream-batch across that depth.

## When SB helps us

The free 1-step inference + B=16 batched dispatch already hits
32.66 FPS. The bottleneck is TE batched encode (~233ms at B=8) plus
the SM-saturated UNet path. SB amortizes:

- TE encode runs once per K-frame stream batch, freeing it from
  per-frame critical path
- VAE encode/decode same amortization
- UNet still does K denoise calls but on K different frames
  simultaneously (same B as before)

**Theoretical gain**: ~K× on TE-bound path. With K=2, SB makes the TE
amortize over twice as many output frames. With our current TE wall
time of ~50% of frame budget, K=2 would reduce TE/frame to 25% →
adding ~8 FPS.

K=3 would push to ~17% → adding another ~5 FPS.

**Realistic target**: from 32.66 → 45-55 FPS at K=2-3 with SB.

## Design

### Pipeline state

Replace single SharedState with a queue of K SharedStates (one per
in-flight frame), tagged by their current denoise step:

```
queue = [
    (frame_t,     step=0),  # entering
    (frame_{t-1}, step=1),  # mid-trajectory
    (frame_{t-2}, step=2),  # exiting
]
```

Each step:
1. Take all K frames from the queue
2. UNet forward: predict v for all K frames at their respective
   timesteps (this is one batched UNet call with B=K*16 if all 16
   frames per batch are at the same step)
3. Update each frame's latent via Euler step with its scheduler
4. Increment step counter for each frame
5. Emit any frame at step >= K_total (write to output)
6. Pull next frame from input, add at step=0

### Sharing TE/VAE

TE encodes the input image+prompt for each NEW frame. Once per K-frame
stream batch (i.e., when a fresh frame enters at step=0).

VAE encodes each new frame's input image (once per frame entry).

VAE decode emits each finished frame (once per frame exit).

### Batched UNet

Within a batch (B=16), if all 16 input frames share the same step,
run them as a B=16 batched UNet call. If they're at different steps,
need to either:
(a) Pad to common step, run, take valid outputs (wasteful)
(b) Group by step and run K separate B=16 calls (loses some batching
    benefit but cleaner)
(c) Mix: run one big B=K*16 call where each frame has its own
    timestep (this requires the UNet to accept a per-batch-element
    timestep, which DreamLite does support)

Option (c) is the cleanest. Verify that
`pipeline.unet(latents, t=timestep_tensor, ...)` works with
timestep_tensor of shape (B,) — likely yes.

### Fixed-pace mode

Initial implementation: assume a fixed K_total and B (say K=2, B=16).
Frames enter and exit deterministically. Steady state has B=16 frames
at each of the K=2 steps (so 32 total in flight).

Later: support dynamic K (different steps for different streams).

## Implementation plan

1. **Verify UNet accepts per-element timestep** (~30 min): inspect
   `dreamlite/models/unets/unet_2d_condition_mobile.py forward()`
   to confirm it accepts a `timestep` tensor of shape (B,) or only
   scalar.

2. **Stream Batch worker** (~3 hours):
   - New class `StreamBatchedEditWorker` in
     `src/dreamlite_stream/workers/stream_batched.py`
   - Manages a queue of K (SharedState, ref_latents, step, frame_idx)
     entries
   - Each call: pulls K frames into one batched UNet call, advances
     each one step, emits expired ones

3. **Integration test** (~1 hour): launch on a single DAVIS sequence,
   verify output correctness against the K=1 champion.

4. **Benchmark** (~30 min): measure FPS for K=1, 2, 3 on
   blackswan/dance-twirl. Compare to champion.

5. **Full DAVIS-9 eval** (~1 hour): run with eval_lcm_lora-style
   script (saves mp4, computes Sobel/HF/LPIPS).

6. **Paper integration** (~30 min): add Stream Batch row to Table 1,
   add §4.6 "Stream Batch as further amortization" or add to ablations.

**Total**: ~6-7 hours including buffer.

## Risk register

- **Per-element timestep**: if the UNet doesn't accept a per-batch
  timestep tensor, we'd need to pad or run separately. Run separately
  is the obvious fallback (loses ~30% theoretical gain but still
  positive).

- **LLLite + Stream Batch interaction**: LLLite cond is per-frame
  (warped_prev). With K different frames at K different steps, each
  has its own cond. The compile-friendly LLLite already supports
  per-batch-element cond_emb, so this should "just work" — but verify
  in test #3.

- **LCM scheduler behavior at non-power-of-2 K**: scheduler chooses
  K timesteps from 1000-step trajectory. For K=2, [999, 0]; K=3,
  [999, 500, 0]. The K=3 case isn't standard — verify scheduler
  produces sensible timesteps.

- **VRAM**: at B=16 × K=2 = 32 in-flight latent tensors, peak VRAM
  doubles vs single-step champion (which already hit 24 GB at
  champion_b16_nf4). May need to reduce B to 8 for K=2.

## Decision: when to stop

If K=2 produces less than 5 FPS gain or breaks quality (visible
flicker / regression on Sobel/HF/LPIPS), stop. The free-1-step path
is already near the TE+UNet ceiling; SB may not have headroom on this
specific stack.

## Pre-flight checklist (before implementation)

- [ ] User confirms direction
- [ ] GPU available (current task block)
- [x] **Confirm `pipeline.unet.forward()` per-element timestep support** —
      VERIFIED. `unet_2d_condition_mobile.py:get_time_embed()` line 980:
      `timesteps = timesteps.expand(sample.shape[0])`. A pre-shaped
      `(B,)` tensor of per-element timesteps is preserved through
      `.expand(B)` (no-op when shape already matches). Pass
      `torch.tensor([t1, t2, ..., tB], device=cuda)` and each batch
      element gets its own scheduler step.
- [ ] Confirm scheduler.set_timesteps(K=2 or 3) produces valid timesteps
- [ ] Decide K and B (likely K=2, B=8 to fit VRAM)
