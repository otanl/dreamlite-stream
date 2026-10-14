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

RIFE interpolation (`demo_camera.py --interp_method rife`) needs only the v4.25 checkpoint at `data/rife/v4.25/flownet.pkl` (gitignored; from the download linked in hzwer/Practical-RIFE). No checkout of that repo is needed; the network is defined in `rife.py`.

Most scripts prepend `src/`, `../dreamlite` and `../dreamlite-lllite/src` to `sys.path` themselves, so they run from the repo root without `pip install -e .`.

Three interpreters on this machine can run the code, and they are not interchangeable. Say which one produced a number.

| Interpreter | Stack | Use for |
|---|---|---|
| WSL Ubuntu `~/projects/venv/bin/python` (3.10) | torch 2.12.0+cu126, transformers 4.57, Linux triton 3.7, dreamlite-lllite installed, TensorRT 10.16.1 and onnx (added 2026-10-14) | Where `out/champion/results.jsonl` was produced (its paths are `/mnt/f/...`); the live demo also runs here (WSLg, NDI). WSL's system `python3` cannot run the pipeline (its transformers predates Qwen3-VL) |
| Windows bare `python` = `C:\Python314` | torch 2.11.0+cu128, transformers 5.2, diffusers 0.39 dev, triton-windows 3.7, sageattention, gemlite, hqq; no TensorRT | Current Windows work; the notes call this the main env |
| `F:/work/dreamlite/.venv/Scripts/python.exe` (3.10) | torch 2.6.0+cu124, transformers 4.57, TensorRT 10.16, onnxruntime-gpu, opencv headless (no window) | Anything TensorRT / ONNX: `build_trt_*.py`, `--trt_engine`, `--rife_engine` |

A TensorRT engine only loads in the TensorRT version that built it, on the platform that built it. Engines under `out/trt/` belong to the Windows TensorRT venv and those under `out/trt_wsl/` to WSL; the spec file records both and a mismatch is refused. `C:\Python314` can only take TensorRT 11.1 or later (first release with cp314 wheels; strongly typed networks only, which is what the build scripts produce).

The interpreters differ in transformers version, and that changes what the text encoder returns: DreamLite conditions on Qwen3-VL's `hidden_states[-1]`, which in transformers 4.57.3 (the version upstream pins) is the last decoder layer's output *before* the language model's final norm, and in transformers 5 is the tensor *after* it. After the UNet's prompt projection the two differ by about 70%, and the pictures differ accordingly (4-step outputs of the two interpreters at 768: LPIPS 0.136). `pipeline_ops.prompt_hidden_state` reads the pre-norm tensor on either version and the in-repo encode paths use it, so the interpreters now agree (embeddings to about 2%; 4-step outputs at 640 to LPIPS 0.012, against 0.150 before). Not covered: upstream's own `pipeline.__call__` / `encode_prompt`, and the scripts that call the encoder themselves (`champion_eval_token_pruning.py`, `measure_te_batch_scaling.py`, `test_idpruner_qwen3vl.py`, `test_vit_compile_qwen3vl.py`); run those with 4.57.3.

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

# Live webcam / NDI demo (lean path at 640x640)
$PY scripts/demo_camera.py --prompt "..." --lean --size 640 --compile_mode reduce-overhead
# ... at 768 with the TensorRT UNet (one-step LoRA inside) and RIFE x2 on the display side
$PY scripts/demo_camera.py --prompt "..." --lean --size 768 --trt_engine out/trt/unet_768_b8_lora.engine \n    --interp_factor 2 --interp_method rife --rife_engine out/trt/rife_768_b8.engine

# Loop throughput and latency; --lean and --size select the operating point
$PY scripts/sustained_throughput_test.py --batch_size 8 --lean --size 640 --out_json out/sustained_lean_640.json

# Training: pairs -> adapter (v3 recipe is recorded in runs/temporal_lllite_v3/args.json)
$PY scripts/generate_temporal_pairs.py ...   # writes data/temporal_pairs_*/
$PY scripts/train_temporal_lllite.py --pairs_dir data/... --out_dir runs/...

# One-step LoRA for the lean path (one per resolution) -> runs/onestep_lora_<size>/
$PY scripts/train_onestep_lora.py --size 768
$PY scripts/train_onestep_lora.py --size 768 --eval_only      # re-score an existing one
$PY scripts/champion_eval.py --lean --size 768 --lllite_multiplier 0 \
    --onestep_lora runs/onestep_lora_768/onestep_lora.safetensors

# TensorRT (interpreter with tensorrt): UNet engine with the LoRA merged in, RIFE engine
$PY scripts/build_trt_unet.py --size 768 --batch_size 8 --lllite_multiplier 0 \
    --onestep_lora runs/onestep_lora_768/onestep_lora.safetensors   # -> out/trt/unet_768_b8_lora.engine
$PY scripts/build_trt_rife.py --size 768 --batch_size 8             # -> out/trt/rife_768_b8.engine
$PY scripts/sustained_throughput_test.py --lean --size 768 --batch_size 8 \
    --trt_engine out/trt/unet_768_b8_lora.engine
```

Tests are standalone scripts with a `__main__` block, not pytest. `scripts/test_idpruner_algorithm.py`, `scripts/test_lean_options.py`, `scripts/test_onestep_lora.py` and `scripts/test_rife_interp.py` run on CPU without weights; the other `scripts/test_*.py` load the model on CUDA, and the `test_trt_*` ones also need an engine from the older TensorRT attempt.

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

### Lean path

`dreamlite_stream/lean.py` defines an opt-in operating point that leaves only the UNet and the VAE in the per-batch path, and the CLI plumbing for it (`--lean` on `champion_eval.py`, `sustained_throughput_test.py`, `demo_camera.py`). It maps to worker fields that also work one at a time:

- `te_refresh_every=0` (+ `te_batch_one`): prompt embedding encoded once, reused until `reset_sequence()` or a forced refresh.
- `cond_refresh_every=0`: LLLite cond set on the first batch and held; no flow, no cond CNN, no previous-frame tracking.
- `shared_noise`: one `(1, C, h, w)` draw expanded to every frame of every batch.
- `fast_vae` (+ `vae_compile`): a private fp16 / channels_last deep copy of the VAE with normalisation and uint8 quantisation on the GPU. `pipeline.vae` is not modified, so other workers sharing the pipeline are unaffected. Frames may be PIL images or uint8 arrays; `output_type="np"` returns arrays.

The default path is unchanged and is what the paper tables use. Lean is a different operating point (the prompt embedding no longer sees every frame), not a faster route to the same numbers. A worker reused across unrelated clips must call `reset_sequence()` when any of this state is held; `lean.holds_clip_state(args)` says when. `scripts/test_lean_options.py` covers the option logic on CPU.

### One-step LoRA

On the lean path the K-step output is a deterministic function of the input frame (shared noise, prompt embedding fixed per clip), so K steps can be distilled into one by paired regression. `scripts/train_onestep_lora.py` does that (Huber on the latent + LPIPS on decoded crops, EMA, teacher targets cached under `data/onestep_targets_<size>/`); `onestep_lora.py` holds the training wrappers and `merge`, which adds the deltas to the UNet weights in place. `lean.apply_onestep_lora` merges before the worker compiles, so the run-time graph is an ordinary UNet.

- A LoRA is tied to its resolution and to one init-noise tensor, which is stored in the file (`init_noise`) and handed to the worker as `init_noise` (`lean_worker_kwargs`); a TensorRT engine keeps a copy next to it (`.noise.safetensors`). The seed is not enough: a seeded bf16 `torch.randn` on the CPU gives one tensor under torch 2.6 / 2.11 on Windows and an unrelated one under torch 2.12 in WSL, while a float32 draw is the same everywhere. `shared_noise` therefore draws in float32 (`make_init_noise(draw_dtype=...)`); the default path keeps its per-batch bf16 draw, so its pictures still depend on the torch build. `--init_noise FILE` runs anything on a stored tensor, which is how a base-model reference for a LoRA has to be made. `compatibility_notes` reports mismatches at load. It is trained on style prompts: on prompts that restructure the scene it stays close to the source frame, so the edit comes out weaker than with more steps.
- Evaluate against the *base* UNet's K-step output. Anything that builds a reference must run before wrappers are attached or after they are detached; `final_eval` in the training script is ordered that way.
- With a LoRA merged the LLLite adapter adds little (DAVIS-10 warp error with / without: 16.12 / 16.19 at 768, 14.86 / 15.31 at 640) and moves the output away from the 4-step reference; run LoRA configs with `--lllite_multiplier 0`. Multiplier 0 skips `apply_lllite` altogether in the three entry scripts (`controller` is `None`).
- Rounding the merged weights into bf16 leaves about a quarter of the delta's norm as rounding error per element, but the output is unaffected (LPIPS to the reference identical to three decimals before and after the merge).

### TensorRT backend

`scripts/build_trt_unet.py` exports the UNet to an fp16 ONNX graph and builds a strongly typed engine; `trt_engine.py::TRTEngineUNet` stands in for `pipeline.unet` (`lean.apply_trt_engine`, flag `--trt_engine`). The worker skips `torch.compile` for it (`is_trt_engine`).

- The prompt projection `encoder_hid_proj` (Linear + RMSNorm) stays outside the engine and runs in PyTorch fp32. Raw Qwen3-VL embeddings reach about 1100 in magnitude and the RMSNorm squares them, which overflows fp16; this is what broke the earlier TensorRT attempt (`trt_unet.py`, `trt_unet_lllite.py`, kept for the paper's negative result).
- The one-step LoRA is merged before export. The adapter, if built in, has its per-hook cond embeddings as engine inputs read from the controller's `_cond_emb_buf`; its multiplier is a constant of the engine.
- An engine is fixed to one size, one batch size and one TensorRT version. `<engine>.json` (`EngineSpec`) records what it was built from; `apply_trt_engine` refuses a mismatching run, and with no explicit `--lllite_multiplier` the spec supplies it.
- Stage 1 (export) also writes `<name>.check.safetensors` with real inputs and the fp32 PyTorch output; stage 2 (build) needs only tensorrt and those files, so the two stages can run under different interpreters.

### RIFE interpolation

`rife.py` holds the RIFE v4.25 network (inference path, parameter names matching the Practical-RIFE checkpoint) and `RIFEInterpolator`, which turns a batch of N output frames into N * factor display frames in one network call. Each frame is encoded once, the last frame is carried so the pair across a batch boundary is interpolated too, and warp coordinates stay fp32 while the convolutions run fp16. Backends: PyTorch, or an engine from `scripts/build_trt_rife.py` (`--rife_engine`). It runs on the demo's display thread.

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

`trt_unet.py` / `trt_unet_lllite.py` (the first TensorRT attempt, reported as a negative result; superseded by `trt_engine.py`), `te_pruning.py` (IDPruner; `HAWKHook` is a placeholder), `sage_attn.py`, `te_vit_compile.py`, and `distillation_losses.py` (LCM-LoRA loss presets B3/O1/O2/O3 for follow-up work) are scaffolding or side studies. `frame_interp.py` (linear blends, temporal blend) and `rife.py` are display-side only.

## Layout conventions

- `scripts/*.py` is the paper's reproduction surface. `scripts/dev/` is superseded history kept for the record; do not build on it.
- `scripts/mobile_o/` holds versioned copies of scripts that execute from `F:\work\Mobile-O` with its own `.venv-mobileo`. The working copies live there; re-sync here before committing.
- `scripts/vast_setup/` bootstraps rented GPU hosts.
- `out/`, `data/`, `runs/`, `notes/`, `figures_for_paper/` and `hf_upload/` are gitignored but present locally. Only `runs/temporal_lllite_v3/args.json` is tracked from `runs/`. Docstrings cite design notes under `notes/` (`demo_tuning.md`, `roadmap.md`, `te_pruning_integration.md`, `sageattn3_integration.md`) that exist only on this machine; `notes/` also holds credentials.
- Weights are never committed. The v3 adapter is expected at `runs/temporal_lllite_v3/temporal_lllite_step001440.safetensors`. Code is Apache-2.0; adapter weights inherit DreamLite's CC BY-NC 4.0 (see `ATTRIBUTION.md`, `USE_POLICY.md`).
- The v3 adapter attaches 38 hooks with `--lllite_blocks down_blocks` and 108 with all blocks; the `[lllite] N hooks` line printed at load is the check.
