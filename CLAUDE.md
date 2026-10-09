# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

Converse with the user in Japanese (会話は日本語で). Everything written into the repo — code, comments, docstrings, commit messages, README — stays in English; this is the public companion artefact of arXiv:2606.05981.

## What this is

Research code for video-rate streaming stylization: an inference runtime, training scripts and the evaluation harness around the DreamLite-mobile UNet + Qwen3-VL text encoder (TE) + a Temporal LLLite adapter. There is no CI, no linter config and no pytest suite; correctness is checked by running experiment scripts and reading their JSONL output.

## Environment

The repo does not run standalone. It expects sibling checkouts next to it:

- `../dreamlite` — upstream ByteVisionLab/DreamLite. Provides `dreamlite.DreamLiteMobilePipeline`; weights at `../dreamlite/models/DreamLite-mobile`.
- `../dreamlite-lllite` — provides `dreamlite_lllite.apply_lllite` / `LLLiteController` (the compile-friendly LLLite module lives there, not here).
- `../Practical-RIFE` — optional, only for `demo_camera.py --interp_method rife`.

Most scripts prepend `src/`, `../dreamlite` and `../dreamlite-lllite/src` to `sys.path` themselves, so they run from the repo root without `pip install -e .`.

Three interpreters on this machine can run the code, and they are not interchangeable. Say which one produced a number.

| Interpreter | Stack | Use for |
|---|---|---|
| WSL Ubuntu `~/projects/venv/bin/python` (3.10) | torch 2.12.0+cu126, transformers 4.57, Linux triton 3.7, dreamlite-lllite installed | Where `out/champion/results.jsonl` was produced (its paths are `/mnt/f/...`); the live demo also runs here (WSLg, NDI). WSL's system `python3` cannot run the pipeline (its transformers predates Qwen3-VL) |
| Windows bare `python` = `C:\Python314` | torch 2.11.0+cu128, transformers 5.2, diffusers 0.39 dev, triton-windows 3.7, sageattention, gemlite, hqq; no TensorRT | Current Windows work; the notes call this the main env |
| `F:/work/dreamlite/.venv/Scripts/python.exe` (3.10) | torch 2.6.0+cu124, transformers 4.57, TensorRT 10.16, onnxruntime-gpu | Anything TensorRT / ONNX |

The first `import torch` under `C:\Python314` can take over a minute; allow for it in command timeouts.

Source files contain non-ASCII characters (em dashes, `×`, `§`). The Windows console codec is cp932, so tooling that prints source text or docstrings needs `PYTHONUTF8=1`.

## Commands

```bash
PY=python   # or one of the interpreters above

# Champion config on the 10 tracked DAVIS clips -> out/champion/results.jsonl
$PY scripts/champion_eval.py
$PY scripts/champion_eval.py --sequences blackswan --max_frames 64   # one clip

# Single clip
$PY scripts/infer_temporal_lllite.py --input assets/davis_mp4/blackswan.mp4 \
    --output out/demo.mp4 --prompt "transfer this to oil painting style, vibrant colors"

# Live webcam / NDI demo
$PY scripts/demo_camera.py --prompt "..." --cond_refresh_every 4 --te_refresh_every 16 \
    --te_batch_one --fixed_noise --compile_mode reduce-overhead

# Training: pairs -> adapter (v3 recipe is recorded in runs/temporal_lllite_v3/args.json)
$PY scripts/generate_temporal_pairs.py ...   # writes data/temporal_pairs_*/
$PY scripts/train_temporal_lllite.py --pairs_dir data/... --out_dir runs/...
```

Tests are standalone scripts with a `__main__` block, not pytest. `scripts/test_idpruner_algorithm.py` is the only one that runs on CPU without weights; the other `scripts/test_*.py` load the model on CUDA, and the `test_trt_*` ones also need a built TensorRT engine.

```bash
$PY scripts/test_idpruner_algorithm.py
# one test function
$PY -c "import sys; sys.path.insert(0,'scripts'); import test_idpruner_algorithm as t; t.test_prune_and_pad_shape_stability()"
```

The script for each paper table and figure is listed in README.md ("Reproducing paper tables and figures").

## Architecture

### The problem the runtime is shaped around

DreamLite's edit mode is a diptych: the noise latent and the reference-frame latent are concatenated along width (`dim=3`), and the UNet output is sliced back to the left half. The prompt is wrapped in an `[Edit]: A diptych ...` template and encoded by Qwen3-VL *together with the frame*, so prompt embeddings are image-dependent and cannot simply be cached across frames. That per-frame TE forward is the bottleneck; the UNet is compiled and already saturated. Almost every design choice here hides or amortises TE cost.

### Layers

1. `pipeline_ops.py` — stateless functions over the pipeline (encode prompt, VAE encode/decode, `denoise`, timesteps). They bypass `pipeline.__call__`, which forces 1024 px for edits and offers no way to reuse embeddings, feed precomputed latents, or time stages separately.
2. `workers/` — dataclasses that own a pipeline plus a `SharedState` (resolution, step count, prompt, frame index):
   - `BatchedEditWorker` is what the paper numbers and the demo use.
   - `EditWorker` is the per-frame predecessor; `runtime.py` (`run_video`, `run_video_pipelined`) drives only this one.
   - `KeyframeWorker` (`[Generate]`-mode T2I) and `SpeculativeEditWorker` (flow-thresholded frame skipping) are earlier explorations.
3. `scripts/` — each script is a self-contained experiment: load pipeline, `apply_lllite`, build a worker, write `out/<name>/results.jsonl`. There is no shared batched driver; the loop is written inline per script. `champion_eval.py::run_sequence` is the canonical one to copy.

### BatchedEditWorker concurrency

- **Side CUDA stream.** `prefetch_batch(next)` issues TE + VAE-encode on `_side_stream` and returns at once with a `PrefetchedBatch` carrying a `done` CUDA event. `step_batch_with_prefetch(cur, pf)` waits on that event, then runs denoise + VAE-decode on the default stream. Callers must kick the next batch's prefetch *before* stepping the current one, or nothing overlaps.
- **CPU pool.** Farneback flow + warp for the LLLite cond image runs in a thread pool (`_maybe_kick_cond_flow`) and is awaited inside the step.
- **Demo-only split.** `prefetch_cpu_only` runs on a background thread and does the TE forward on a separate `_te_side_stream`, so the `.item()` sync inside Qwen3-VL blocks that thread rather than the pipeline thread. `prefetch_gpu_kick` then runs on the pipeline thread. Cross-stream reads of the TE cache are ordered by `_last_cache_te_ev`.
- **Amortisation knobs.** `cond_refresh_every` (reuse the LLLite cond embedding for N batches), `te_refresh_every` (reuse TE embeddings), `te_batch_one` (encode one frame and broadcast).

### Temporal LLLite

`apply_lllite(pipeline.unet, inference_mode=True, max_batch_size=B, block_filter=[...])` patches attention Linear layers and must be called before `torch.compile` so the hooks are inside the compiled graph. The cond image passed to `controller.set_cond_image` is the previous batch's last *decoded output*, warped by optical flow computed on the *input* frames (`flow.py`); the first batch uses the input frames themselves. `metrics.py` and `output_blend.py` use the same flow-on-input convention, so the headline `warp_err` metric and the conditioning agree on what counts as motion.

### Invariants that break compilation when violated

The UNet is compiled with `mode="reduce-overhead", dynamic=False`, i.e. captured as CUDA graphs. Inputs must keep the same shape, memory layout and calling thread:

- Batch size is fixed. Drop a partial final batch rather than running it, and keep LLLite `max_batch_size` equal to the worker's `batch_size`.
- Prompt sequence length is fixed by `padding="max_length", max_length=256`; `drop_idx=64` slices off the system template. The template, decoration and both constants are duplicated in `pipeline_ops.encode_prompt_edit_batch` and `BatchedEditWorker._PROMPT_*` and must stay identical.
- Cached TE tensors are made `.contiguous()` and are `.clone()`d per call so the graph sees a fresh, same-layout input. The `te_batch_one` branch is repeated in three prefetch paths and must match across them.
- `prefetch_gpu_kick` and `step_batch_with_prefetch` must run on the same thread.
- No host sync in the hot path; the padded-slice trick in `encode_prompt_edit_batch` exists to avoid one.
- `_enable_compile` is a no-op if the UNet is already wrapped (`_orig_mod`). Build one worker and reuse it across sequences, resetting `state.frame_idx`, `_last_prev_decoded` and `_last_prev_input_gray` between them; a worker per sequence exhausts VRAM.
- Long-running scripts raise `torch._dynamo.config.cache_size_limit` at import time to avoid recompile thrashing.

### What the printed numbers measure

- There are two throughput figures and they differ by about 2x on the shipped path. Step-only fps times `step_batch_with_prefetch`; loop fps also counts the `prefetch_batch(next)` call that runs on the same thread just before it (tokenisation, PIL preprocessing, and a text-encoder dispatch that blocks on host syncs). `BatchTiming.total_ms` is the step, `prefetch_ms` the prefetch call, `loop_ms` their sum. `champion_eval.py` (`fps` / `fps_loop`) and `sustained_throughput_test.py` report both; the other eval scripts still derive fps from `total_ms` alone, so theirs is step-only. Quote loop fps when judging a change.
- `fixed_noise=True` reuses one `(B, C, h, w)` draw for every batch, so each batch slot keeps its own pattern and neighbouring frames still get different noise. It is not one pattern shared by all frames.
- With `cond_refresh_every=8` and 64-frame clips, `champion_eval.py` sets the LLLite cond exactly once per clip, from the first batch's own input frames (there is no previous output yet), so those runs never feed a flow-warped previous output.

### Modules outside the main path

`trt_unet.py` / `trt_unet_lllite.py` (TensorRT, reported as a negative result), `te_pruning.py` (IDPruner; `HAWKHook` is a placeholder), `sage_attn.py`, `te_vit_compile.py`, and `distillation_losses.py` (LCM-LoRA loss presets B3/O1/O2/O3 for follow-up work) are scaffolding or side studies. `frame_interp.py` is display-side only and does not affect pipeline throughput.

## Layout conventions

- `scripts/*.py` is the paper's reproduction surface. `scripts/dev/` is superseded history kept for the record; do not build on it.
- `scripts/mobile_o/` holds versioned copies of scripts that execute from `F:\work\Mobile-O` with its own `.venv-mobileo`. The working copies live there; re-sync here before committing.
- `scripts/vast_setup/` bootstraps rented GPU hosts.
- `out/`, `data/`, `runs/`, `notes/`, `figures_for_paper/` and `hf_upload/` are gitignored but present locally. Only `runs/temporal_lllite_v3/args.json` is tracked from `runs/`. Docstrings cite design notes under `notes/` (`demo_tuning.md`, `roadmap.md`, `te_pruning_integration.md`, `sageattn3_integration.md`) that exist only on this machine; `notes/` also holds credentials.
- Weights are never committed. The v3 adapter is expected at `runs/temporal_lllite_v3/temporal_lllite_step001440.safetensors`. Code is Apache-2.0; adapter weights inherit DreamLite's CC BY-NC 4.0 (see `ATTRIBUTION.md`, `USE_POLICY.md`).
- The v3 adapter attaches 38 hooks with `--lllite_blocks down_blocks` and 108 with all blocks; the `[lllite] N hooks` line printed at load is the check.
