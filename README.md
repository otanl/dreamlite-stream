# dreamlite-stream

Video-rate streaming stylization on the smallest vision-aware
MLLM-conditioned edit-diffusion stack.

> ⚠️ **Non-commercial, research-only.** The trained adapter
> weights in this repository inherit DreamLite-mobile's CC BY-NC 4.0
> weight licence and the upstream use policy
> (no NSFW / violent / discriminatory / illegal content, no
> commercial use, no malicious use). See [`USE_POLICY.md`](USE_POLICY.md)
> for the verbatim upstream notice and what it means in practice.

This repository accompanies the preprint **_Video-Rate Streaming
Stylization on a Vision-Aware MLLM-Conditioned Edit Diffusion:
Asymmetric Batched Inference on a Distilled UNet + MLLM Text Encoder_**
([arXiv:2606.05981](https://arxiv.org/abs/2606.05981)) and contains
the inference runtime, training scripts, evaluation harness, and a
trained Temporal LLLite adapter (v3) used to produce the paper's
tables and figures.

## Demo

<video src="https://github.com/user-attachments/assets/ef31f68a-bc6a-4e14-b2b9-6a60303db30f" controls width="640">
  See <a href="assets/demo.mp4">assets/demo.mp4</a> for the live-demo sample.
</video>

`scripts/demo_camera.py` stylizing a
[Pexels](https://www.pexels.com/) stock clip on a 3090 Ti at
512×512. Observed end-to-end fps is slightly below the paper's
step-only sustained number due to display pacing and frame-drop
bookkeeping in the demo path.

## What is in this repository

- A side-stream / main-stream CUDA pipeline that hides the Qwen3-VL
  text-encoder cost behind the compiled DreamLite-mobile UNet
  (see `src/dreamlite_stream/workers/batched_edit.py`).
- A **compile-friendly** reformulation of the Temporal LLLite
  adapter (lives in the sibling `dreamlite-lllite` repo) that lets
  `torch.compile` fold the whole UNet+adapter stack into a single
  graph.
- A *periodic cond-refresh + hook-subset* schedule (`down_blocks`,
  refresh every 8 batches) that amortises the LLLite conditioning
  CNN over multiple batches.
- The full evaluation harness used to produce every numerical row
  in the paper (Section "Reproducing paper tables and figures" below).
- Beyond the paper, all opt-in: the lean path, a one-step LoRA that
  brings one denoising step close to the multi-step output, a TensorRT
  backend for the UNet, and batched RIFE interpolation on the display
  side (the last four sections before "Reproducing paper tables").

The base model and text encoder are not redistributed here. We
build on the **DreamLite-mobile** UNet
([ByteVisionLab/DreamLite](https://github.com/ByteVisionLab/DreamLite))
and **Qwen3-VL** TE; both are obtained from their upstream
repositories under their respective licenses.

## Hardware requirements

- CUDA 12.6+, an NVIDIA GPU with **≥ 12 GB VRAM**.
- We measured on RTX 3090 Ti, RTX 4090, and RTX 5090 (Blackwell
  requires a torch nightly with `cu128`; see `scripts/vast_setup/`).
- A working `triton` install is required for `torch.compile` mode
  `reduce-overhead`; the host running our 3090 Ti / 4090 / 5090
  numbers used a Linux Python wheel of triton. On Windows hosts
  without `triton`, the eval scripts will fall back to eager mode
  (numerically equivalent, slower).

## Install

```bash
# Sibling repos expected at: ../dreamlite, ../dreamlite-lllite
git clone https://github.com/ByteVisionLab/DreamLite               ../dreamlite
git clone https://github.com/<this-org>/dreamlite-lllite           ../dreamlite-lllite
git clone https://github.com/<this-org>/dreamlite-stream           dreamlite-stream

cd dreamlite-stream
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ../dreamlite-lllite
# DreamLite-mobile weights: follow upstream instructions to obtain
# them, then place them under  ../dreamlite/models/DreamLite-mobile
```

## Quick start

Reproduce the champion configuration on the 10 DAVIS-2017
sequences from Table II of the paper:

```bash
python scripts/champion_eval.py
# Writes:
#   out/champion/champion/<seq>.mp4
#   out/champion/results.jsonl   (per-sequence fps, fps_loop + warp_err)
```

Two rates are reported. `fps` times `step_batch_with_prefetch` alone
(step-only). `fps_loop` also counts the time the same thread spends in
`prefetch_batch` for each batch, so it is the rate a caller of the loop
gets. `scripts/sustained_throughput_test.py` prints both as well.

Single-clip inference:

```bash
python scripts/infer_temporal_lllite.py \
    --input  assets/davis_mp4/blackswan.mp4 \
    --output out/demo.mp4 \
    --prompt "transfer this to oil painting style, vibrant colors"
```

Live webcam / NDI demo (input | stylized output side-by-side, fps overlay):

```bash
# Webcam (uses --camera N for the OpenCV VideoCapture index):
python scripts/demo_camera.py --prompt "transfer this to oil painting style, vibrant colors"

# NDI source (e.g. a TouchDesigner NDI Out named like "...(td)"):
python scripts/demo_camera.py --ndi_source td --prompt "..."

# Recommended config for a smooth live feed: the lean path (see "Lean path
# and resolution" below) at 640x640, about the largest size a 3090 Ti keeps
# near video rate.
python scripts/demo_camera.py \
    --prompt "transfer this to oil painting style, vibrant colors" \
    --lean --size 640 \
    --compile_mode reduce-overhead

# With a one-step LoRA and the TensorRT engines built (see "One-step LoRA",
# "TensorRT backend" and "Frame interpolation" below), 768x768 runs at about
# 23 fps and is displayed at twice that:
python scripts/demo_camera.py     --prompt "transfer this to oil painting style, vibrant colors"     --lean --size 768 --trt_engine out/trt/unet_768_b8_lora.engine     --interp_factor 2 --interp_method rife --rife_engine out/trt/rife_768_b8.engine

# The earlier amortised config (per-frame work kept, refreshed on a schedule):
python scripts/demo_camera.py \
    --prompt "transfer this to oil painting style, vibrant colors" \
    --cond_refresh_every 4 --te_refresh_every 16 \
    --te_batch_one --fixed_noise \
    --compile_mode reduce-overhead
```

Window keys (focus the demo window): `q` quit · `space` pause/resume ·
`s` save current output · `p` print prompt-input hint.

**Live prompt changes** — type a new prompt in the launching terminal
(look for `prompt>`) and press Enter. The pipeline does not pause; the
new style appears ~1–2 iters later (~600 ms). Empty Enter = keep
current.

The demo uses the same `BatchedEditWorker` as `champion_eval.py` plus
a dedicated bg thread that runs Qwen3-VL on its own CUDA stream so the
``.item()`` sync inside the vision encoder doesn't stall the main
UNet step. At source 30 fps with B=8, end-to-end camera→display latency
is ~700–900 ms (one batch buffered on the input side, one in the
pipeline, one frame paced on the display side). The paper's sustained
number is step-only (`fps` vs. `fps_loop` above); the demo's wall-clock
fps also includes the prefetch work, display pacing and frame-drop
bookkeeping.

**RIFE setup** (only needed if you pass `--interp_method rife`): download
RIFE v4.25 from the link in the
[Practical-RIFE](https://github.com/hzwer/Practical-RIFE) README and put
its `train_log/flownet.pkl` at `data/rife/v4.25/flownet.pkl` (or pass
`--rife_weights`). No checkout of that repository is needed. See "Frame
interpolation" below.

```bash
python scripts/demo_camera.py --prompt "..." --lean --size 640 \
    --interp_factor 2 --interp_method rife
```

Useful tuning flags (`--help` for the full list):

| Flag | What it does |
|---|---|
| `--lean` | Preset for the reduced per-batch path: `--te_refresh_every 0 --te_batch_one --cond_refresh_every 0 --shared_noise --fast_vae --vae_compile`. See "Lean path and resolution" below. |
| `--size N` | Square working resolution (default 512). The base model is trained to edit at 1024; 640 and 768 keep much more structure than 512 at about 1.6x and 2.5x the cost. |
| `--te_refresh_every N` | Cache Q3-VL prompt_embeds across N batches. Hides the ~250 ms TE-forward cost on the cached majority of iters; trades static-prompt image-awareness for throughput. `0` = encode once and refresh only when the prompt changes. |
| `--te_batch_one` | When refreshing, run Q3-VL on a single representative frame and broadcast — multimodal sequence shortens ~B×. Only sensible when paired with `--te_refresh_every` other than 1. |
| `--cond_refresh_every N` | Same amortisation for the LLLite temporal cond (Farneback flow + warp). `0` = set it once and hold it (no flow, no cond CNN per batch). |
| `--shared_noise` | One init-noise draw for every frame of every batch. A static input then gives the same output every frame; this is the largest single reduction in flicker. |
| `--fixed_noise` | Reuse the same init-noise tensor for every batch. Each of the B slots keeps its own pattern, so neighbouring frames still differ; prefer `--shared_noise`. Demo-only: the paper's evaluation scripts draw new noise per batch. |
| `--fast_vae`, `--vae_compile` | Run the VAE in fp16 / channels_last with normalisation on the GPU instead of the PIL image processor, optionally `torch.compile`d (adds a compile at start-up). |
| `--onestep_lora FILE` | Merge a one-step LoRA into the UNet so one step lands close to the 4-step output at no run-time cost. Tied to one `--size` and to the init-noise tensor stored in the file; use with `--lean`. See "One-step LoRA" below. |
| `--init_noise FILE` | Use the init-noise tensor stored in FILE (a one-step LoRA, or the `.noise.safetensors` next to an engine) as the shared noise instead of drawing one from `--seed`. `--onestep_lora` and `--trt_engine` do this by themselves. |
| `--lllite_multiplier F` | Scale of the Temporal LLLite adapter (default 1.0). `0` runs without the adapter: its hooks are not attached. |
| `--quant_te {nf4,fp4,int8}` | Quantise Q3-VL on load (requires `bitsandbytes`). For fitting in lower VRAM; speed-neutral on ≥12 GB hosts. |
| `--compile_mode reduce-overhead` | Best demo throughput; enables `torch.compile`'s CUDA Graph capture (requires `triton`). |
| `--interp_factor N` | Display-side frame interpolation. `2` doubles, `4` quadruples the displayed fps without invoking the model again. Method is chosen by `--interp_method`. Pipeline throughput unchanged. |
| `--interp_method {linear,rife}` | `linear` (default) is a cheap pixel blend; moving objects show as double images. `rife` computes flow-based in-betweens with RIFE v4.25 for the whole batch in one call on the output side (the input side stays linear). Needs `--rife_weights` (default `data/rife/v4.25/flownet.pkl`) or `--rife_engine`. |
| `--rife_engine FILE` | TensorRT engine of the RIFE network from `scripts/build_trt_rife.py`, built for this `--size` and `--batch_size`; about twice as fast as the PyTorch network. |
| `--trt_engine FILE` | Run the UNet as a TensorRT engine from `scripts/build_trt_unet.py` instead of `torch.compile`. See "TensorRT backend" below. |
| `--temporal_blend_alpha F` | Sequential temporal blend on output frames (mix `1-α` of current with `α` of predecessor). Reduces static-region flicker; over-strong values blur fast motion. `0.0` = off (default), `0.3–0.6` is the useful range. |
| `--temporal_blend_warp` | When blending, Farneback-warp the predecessor toward the current frame first. Removes most motion ghosting at the cost of ~5 ms CPU per frame. Pair with `--temporal_blend_alpha > 0`. |

## Lean path and resolution

The default path is the paper's configuration: it re-encodes the prompt
with Qwen3-VL for every frame, refreshes the LLLite cond on a schedule,
draws new init noise per batch, and moves frames through the PIL image
processor. `--lean` (on `demo_camera.py`, `champion_eval.py` and
`sustained_throughput_test.py`) leaves only the UNet and the VAE in the
per-batch path:

| Part of `--lean` | Effect |
|---|---|
| `--te_refresh_every 0 --te_batch_one` | Prompt embedding computed once per clip (once per prompt in the demo), from one frame |
| `--cond_refresh_every 0` | LLLite cond set once and held |
| `--shared_noise` | One init-noise draw shared by every frame |
| `--fast_vae --vae_compile` | fp16 / channels_last compiled VAE, uint8 arrays in and out |

Each flag also works on its own. This is a different operating point, not a
faster route to the paper's numbers: in particular the prompt embedding no
longer sees every frame. The tables in the paper use the default path.

Loop fps (`prefetch + step`) on an RTX 3090 Ti, B=8, one step, measured
with `scripts/sustained_throughput_test.py`:

| Size | default | `--lean` |
|---|---|---|
| 512 | 15.1 | 43.6 |
| 640 | 10.6 | 26.3 |
| 768 | 8.2 | 17.1 |

On the 10 DAVIS clips at 512 (`scripts/champion_eval.py`), `--lean` gives
`fps_loop` 43.9 and warp error 14.75, against 15.1 and 19.94 for the
default path. Warp error is computed at the working resolution, so it is
not comparable across sizes.

```bash
# Throughput and latency at 640x640, batch 4 for lower latency
python scripts/sustained_throughput_test.py --lean --size 640 --batch_size 4

# Quality on DAVIS at 640x640
python scripts/champion_eval.py --lean --size 640 --out_dir out/champion_lean_640
```

The released v3 adapter was trained at 512 but still lowers warp error at
640 and 768. `python scripts/test_lean_options.py` checks the option logic
on CPU without model weights.

## One-step LoRA

On the lean path every frame starts from the same noise tensor and the
prompt embedding is fixed for the clip, so the multi-step output is a
deterministic function of the input frame. That turns step distillation
into paired regression: `scripts/train_onestep_lora.py` fits a LoRA
(rank 32 on the attention and feed-forward Linears) so that **one** step
reproduces the base UNet's 4-step output, with a Huber loss on the latent
and LPIPS on decoded crops. With the defaults it builds 1,848 training
pairs from 40 clips and 8 style prompts and trains for 3,000 steps: about
45 minutes on an RTX 3090 Ti at either 640 or 768, including the teacher
targets (13 GB and 18 GB peak). At run time the LoRA is merged into the
UNet weights, so a step costs what it cost before.

```bash
# Train (teacher targets are cached under data/onestep_targets_<size>/)
python scripts/train_onestep_lora.py --size 768     # -> runs/onestep_lora_768/

# Use
python scripts/champion_eval.py --lean --size 768 --lllite_multiplier 0 \
    --onestep_lora runs/onestep_lora_768/onestep_lora.safetensors
python scripts/demo_camera.py --prompt "..." --lean --size 768 --lllite_multiplier 0 \
    --onestep_lora runs/onestep_lora_768/onestep_lora.safetensors \
    --compile_mode reduce-overhead
```

Held-out clips, every row scored against the base UNet's 4-step output at
the same size (`runs/onestep_lora_<size>/final_eval.json`; 2 clips x 32
frames per prompt). LPIPS: lower is closer to 4 steps. Sobel: sharpness
relative to 4 steps. Edge: agreement of the output's edge map with the
source frame.

| Size | Prompts | Steps | LPIPS | Sobel | Edge | Warp err |
|---|---|---|---|---|---|---|
| 768 | the 8 training styles | 1 | 0.238 | x0.87 | 0.636 | 17.22 |
| | | 1 + LoRA | 0.159 | x0.92 | 0.700 | 15.34 |
| | | 2 | 0.127 | x0.96 | 0.661 | 17.38 |
| | 2 styles not trained on | 1 | 0.220 | x0.88 | 0.733 | 14.38 |
| | | 1 + LoRA | 0.177 | x1.05 | 0.774 | 14.08 |
| | | 2 | 0.106 | x0.98 | 0.749 | 14.73 |
| 640 | the 8 training styles | 1 | 0.260 | x0.79 | 0.642 | 16.32 |
| | | 1 + LoRA | 0.171 | x0.91 | 0.682 | 15.52 |
| | | 2 | 0.141 | x0.94 | 0.658 | 16.91 |
| | 2 styles not trained on | 1 | 0.265 | x0.73 | 0.700 | 13.87 |
| | | 1 + LoRA | 0.207 | x0.96 | 0.723 | 14.44 |
| | | 2 | 0.133 | x0.90 | 0.698 | 15.30 |

One step with the LoRA closes about three quarters of the LPIPS gap between
one and two steps on the styles it was trained on and about 40% of it on
other styles, keeps more of the source structure than either, and has the
lowest warp error on the trained styles. Edit-type prompts are outside its
training set: on four of them at 768 (snow, sunset, night, lego) LPIPS goes
from 0.301 to 0.275 against 0.190 for two steps, and where the multi-step
result restructures the scene, the LoRA's output stays close to the source
frame, i.e. the edit is weaker.

The 10 DAVIS clips (`champion_eval.py --lean`, oil painting prompt, 64
frames per clip), scored from the written mp4s against a 4-step run without
the adapter. Every row runs on the init noise stored in the LoRA file
(`--init_noise` for the rows without the LoRA). Loop fps is from the same
runs (RTX 3090 Ti, Windows, torch 2.6):

| Size | Config | Loop fps | Warp err | Sobel | Edge | LPIPS to 4 steps |
|---|---|---|---|---|---|---|
| 768 | 4 steps, no adapter (reference) | 5.2 | 16.21 | x1.00 | 0.723 | - |
|  | 1 step + adapter (`--lean` as is) | 17.4 | 18.51 | x0.99 | 0.664 | 0.186 |
|  | 1 step, no adapter | 17.9 | 19.07 | x1.01 | 0.664 | 0.188 |
|  | 1 step + LoRA + adapter | 17.4 | 16.12 | x0.92 | 0.722 | 0.126 |
|  | 1 step + LoRA, no adapter | 17.9 | 16.19 | x0.93 | 0.725 | 0.109 |
|  | 1 step + LoRA, no adapter, TensorRT engine | 25.1 | 16.20 | x0.93 | 0.728 | 0.109 |
|  | 2 steps, no adapter | 9.9 | 17.32 | x1.02 | 0.707 | 0.096 |
| 640 | 4 steps, no adapter (reference) | 7.6 | 15.37 | x1.00 | 0.706 | - |
|  | 1 step + adapter (`--lean` as is) | 25.9 | 16.29 | x0.88 | 0.667 | 0.211 |
|  | 1 step, no adapter | 26.8 | 17.15 | x0.94 | 0.661 | 0.210 |
|  | 1 step + LoRA + adapter | 26.0 | 14.86 | x0.89 | 0.700 | 0.138 |
|  | 1 step + LoRA, no adapter | 26.8 | 15.31 | x0.93 | 0.702 | 0.123 |
|  | 1 step + LoRA, no adapter, TensorRT engine | 38.5 | 15.33 | x0.94 | 0.704 | 0.123 |
|  | 2 steps, no adapter | 15.0 | 16.17 | x1.00 | 0.700 | 0.113 |

With the LoRA merged, the Temporal LLLite adapter changes warp error by
-0.07 at 768 and -0.45 at 640 and moves the output away from the 4-step
reference, so the commands above run without it.

Limits. A LoRA is fitted to one resolution and one init-noise tensor and
needs `--shared_noise` (part of `--lean`); `--onestep_lora` prints a note
for each mismatch it can see. The noise tensor is stored in the LoRA file
and used as the shared noise, because a seed does not pin it down: a seeded
bf16 draw is a different tensor on different torch builds (see
"Reproducibility caveats"). To compare against the base model, run the base
model on the same tensor with `--init_noise <the LoRA file>`. The LoRA is
trained on style prompts only, and prompts far from those get a smaller
gain. It is a derivative of the DreamLite-mobile weights (CC BY-NC 4.0) and
is not distributed with this repository.
`python scripts/test_onestep_lora.py` checks attach / merge / restore on
CPU without model weights.

## TensorRT backend

`scripts/build_trt_unet.py` builds a TensorRT engine of the UNet for one
resolution and batch size, and `--trt_engine FILE` (on the same three
scripts as `--lean`) runs it in place of the `torch.compile`d UNet. The
engine contains the merged one-step LoRA and, if built with it, the
Temporal LLLite adapter.

The paper reports its TensorRT attempt as a negative result (fp16 output
drift). The cause is the prompt projection at the UNet's input: raw
Qwen3-VL embeddings reach about 1100 in magnitude and the projection's
RMSNorm squares them, which overflows fp16. The build therefore leaves that
projection (a Linear and an RMSNorm) out of the engine and the runtime
computes it in PyTorch fp32; everything after it has small activations.
With that change the fp16 engine is within 0.7% of the fp32 PyTorch UNet on
real prompt embeddings, and on the 10 DAVIS clips its output matches the
`torch.compile` path (LPIPS 0.01 between the two written videos, the same
warp error), with or without the adapter in the engine.

```bash
# Needs an interpreter with tensorrt. An engine only loads in the
# TensorRT version that built it, on the platform that built it.
python scripts/build_trt_unet.py --size 768 --batch_size 8 --lllite_multiplier 0 \
    --onestep_lora runs/onestep_lora_768/onestep_lora.safetensors
#   -> out/trt/unet_768_b8_lora.engine (+ .json spec, .noise.safetensors,
#      .onnx, .check.safetensors)

python scripts/sustained_throughput_test.py --lean --size 768 --batch_size 8 \
    --trt_engine out/trt/unet_768_b8_lora.engine
python scripts/demo_camera.py --prompt "..." --lean --size 768 \
    --trt_engine out/trt/unet_768_b8_lora.engine
```

Loop fps on an RTX 3090 Ti, B=8, one step, LoRA merged, no adapter
(`sustained_throughput_test.py`; Windows, torch 2.6, TensorRT 10.16):

| Size | `torch.compile` | TensorRT, strongly typed | TensorRT, fp16 (default on TensorRT 10) |
|---|---|---|---|
| 640 | 26.4 | 33.2 | 36.9 |
| 768 | 17.5 | 20.8 | 23.8 |

With the adapter in the engine and no LoRA, 640 goes from 25.6 to 32.5
(strongly typed). In WSL (torch 2.12, same GPU, fp16 engines built there)
the same test gives 36.8 at 640 and 23.9 at 768.

Notes:

- `--precision fp16` lets TensorRT choose fp16 or fp32 per layer and is the
  faster build on TensorRT 10 (0.7% error against fp32, as the bf16
  `torch.compile` path has). `--precision typed` keeps every layer in the
  dtype of the exported fp16 graph (0.4%) and is the only mode TensorRT 11
  offers. A 768 engine takes 5 to 10 minutes to build and about 750 MB.
- The engine fixes `--size` and `--batch_size`; `<engine>.json` records
  them together with the LoRA file, the adapter multiplier, the TensorRT
  version and the platform, and `--trt_engine` refuses a run that does not
  match. Without an explicit `--lllite_multiplier` the engine's value is
  used. The init noise of the built-in LoRA sits next to the engine
  (`.noise.safetensors`) and becomes the run's shared noise.
- A Windows engine does not load under Linux / WSL and the reverse. With
  both sharing one checkout, give each its own folder
  (`--out out/trt_wsl/unet_768_b8_lora.engine`); the build refuses to
  overwrite an engine from the other platform unless `--force` is given.
- The export stage also writes real inputs and the fp32 PyTorch output for
  them (`.check.safetensors`). The build stage needs only `tensorrt` and
  those files, so the two stages can run under different interpreters, and
  it prints the engine's error and time per call when it finishes.

## Frame interpolation

`--interp_factor 2 --interp_method rife` doubles the displayed frame rate
with RIFE v4.25 in-betweens (`src/dreamlite_stream/rife.py`). One network
call covers a whole batch: every frame is encoded once, and the last frame
of a batch is carried over so the pair across the batch boundary is
interpolated too. A linear cross-fade shows a moving subject twice; the
flow-based in-between shows it once, part-way.

```bash
python scripts/build_trt_rife.py --size 768 --batch_size 8      # optional engine
python scripts/demo_camera.py --prompt "..." --lean --size 768 \
    --trt_engine out/trt/unet_768_b8_lora.engine \
    --interp_factor 2 --interp_method rife --rife_engine out/trt/rife_768_b8.engine
```

Time for the 8 in-between frames of one batch on an RTX 3090 Ti:

| Size | PyTorch fp16 | TensorRT |
|---|---|---|
| 640 | 27.4 ms | 12.8 ms |
| 768 | 38.4 ms | 18.1 ms |

The demo loop run headless (a clip fed at 30 fps, window calls replaced by
stubs with exact waits; TensorRT UNet with the one-step LoRA):

| Config | Pipeline fps | Displayed fps |
|---|---|---|
| 768 | 23.1 | 23.1 |
| 768 + RIFE x2 (TensorRT) | 22.0 | 43.9 |
| 768 + RIFE x2 (PyTorch) | 20.6 | 41.0 |
| 640 + RIFE x2 (TensorRT) | 30.0 (the source rate) | 60.0 |

Whether a real window reaches the same displayed rate depends on how
precisely `cv2.waitKey` waits on the host. The engine agrees with the fp32
PyTorch network to 69 dB PSNR. `python scripts/test_rife_interp.py` checks
the batching, the carried frame and the padding on CPU without weights.

## Reproducing paper tables and figures

Table labels below match the arXiv preprint
(Roman-numeral table numbering; content described in case of future
re-numbering).

| Paper artefact (content) | Script | Notes |
|---|---|---|
| Table II — Main DAVIS-2017 results | `scripts/comprehensive_ablation.py` | all rows of the speed pipeline + LLLite + hook subset |
| Table III — Sustained throughput + latency | `scripts/sustained_throughput_test.py` | 480-frame *parkour* loop, p50/p95 latency |
| Table IV — In-pipeline component profile | `scripts/profile_inpipeline.py` | TE/UNet/VAE per-batch wall, side vs. main |
| Table V — Per-component batch scaling | `scripts/measure_te_batch_scaling.py` + `scripts/measure_batch_scaling.py` | TE and U-Net in isolation at B in {1,2,4,8}; sub-linearity per component |
| Table VI — Cond-refresh sweep | `scripts/cond_refresh_sweep_downblocks.py` + `scripts/cond_refresh_spatial_metrics.py` | N in {1,4,8,16} fps/εw + Sobel/HF-FFT/LPIPS-to-N=1 |
| Table VII — Long-sequence drift | `scripts/sustained_throughput_test.py --chunked` | 480-frame *parkour*, per-64-frame chunk std |
| Table VIII — Scene-cut robustness | `scripts/scene_cut_eval.py`, `scripts/scene_cut_lpips.py` | synthetic hard-cut clips |
| Table IX — Transfer evaluation (v5-heldout / DAVIS-19 unused / cross-dataset) | `scripts/eval_heldout_video.py`, `scripts/champion_eval.py --sequences ...`, `scripts/phase3_crossds_local.py` | (a)+(b)+(c) panels of Table IX |
| Table X — Multi-prompt v4 LLLite | `scripts/eval_multiprompt.py` | 5-prompt v4 LLLite |
| Table XI — Hardware scaling (3090 Ti / 4090 / 5090) | `scripts/champion_eval.py` per host | sustained fps headline across GPUs |
| Table XII — LCM-LoRA distillation case study | `scripts/train_lcm_lora.py` + `scripts/champion_eval.py` | v1 blended-teacher target negative result |
| Figure 1 — Smoothing artefact teaser | `scripts/viz_temporal_pair.py` | LCM-LoRA vs. LLLite vs. baseline |
| Figure 2 — Pipeline schematic | — | drawn from Table IV numbers, source SVG in the paper repo |
| Figures 3, 4 — Qualitative grids | `scripts/extract_figure_frames.py` + `build_qualitative_grid_svg.py` + `build_scenecut_grid_svg.py` | regenerate from the saved mp4s |
| Negative result — Token pruning | `scripts/champion_eval_token_pruning.py` | pruned-prompt eval |
| Negative result — TensorRT path | `scripts/export_unet_lllite_onnx.py`, `scripts/build_trt_engine_mixed.py`, `scripts/debug_lllite_trt_drift.py` | LLLite-baked TRT export attempt as reported in the paper; "TensorRT backend" above gives the cause and a working build |

Each script writes its per-row results to `out/<name>/results.jsonl`
next to the stylized mp4s. `out/` is not tracked in this repository
(see `.gitignore`), so the table numbers are regenerated by running
the scripts rather than read from committed logs.

## Released artefacts

- **v3 Temporal LLLite checkpoint** (`temporal_lllite_step001440.safetensors`,
  ~51 MB; 38 attention hooks on the DreamLite-mobile UNet
  `down_blocks` after load-time subset filtering). The weights
  are Adapted Material of DreamLite-mobile and inherit
  DreamLite's CC BY-NC 4.0 weight licence. Mirrors:
    - GitHub release: <https://github.com/otanl/dreamlite-stream/releases/tag/v0.1.0-preprint>
      (asset `temporal_lllite_step001440.safetensors`, SHA-256
      `88082c6bf56770469ad4ecbbca467b315ffcf4b5287fd17733751e2952fee7fc`).
    - HuggingFace: <https://huggingface.co/otnl/dreamlite-stream-temporal-lllite-v3>
      (same file, same SHA-256).
    - Zenodo (code state at submission): <https://doi.org/10.5281/zenodo.20389428>.

  The training-args metadata is committed under
  `runs/temporal_lllite_v3/args.json` so the recipe is fully
  reproducible from the training script
  (`scripts/train_temporal_lllite.py`) given the upstream
  DreamLite-mobile weights.
- **Synthetic scene-cut clips** (Appendix B) under `assets/scene_cut/`.

## Reproducibility caveats

We do not redistribute the DreamLite-mobile UNet weights or the
Qwen3-VL TE weights. The preprint's Reproducibility section documents
what can and cannot be verified without upstream weight access:
the **inference pipeline reformulation**, the
**compile-friendly LLLite module**, the **trained adapter**, the
**benchmark scripts**, and the **TRT export tooling** are all in this
repo. The per-sequence JSONL logs behind the tables are produced by
the benchmark scripts and are not shipped here. The end-to-end
headline numbers require obtaining the upstream gated weights under
their respective licenses.

**Init noise and torch builds.** `torch.randn` with a seeded CPU generator
does not give the same tensor on every torch build when asked for bf16
directly: torch 2.6 and 2.11 on Windows agree with each other, torch 2.12
on Linux returns an unrelated tensor (correlation 0.003), while a float32
draw is the same on all three. The default path draws its noise that way
(bf16, per batch) and is left as it is, so its pictures follow the torch
build. The lean path's shared noise is drawn in float32 and cast, which
gives the same tensor on those three builds (to 1e-7), and a one-step LoRA
stores the tensor it was fitted to. Measured on the 10 DAVIS clips at 768:
the same LoRA config under Windows / torch 2.6 and WSL / torch 2.12 differs
by LPIPS 0.19 when each draws bf16 noise from the seed, and by 0.01 with
the stored tensor.

**transformers version.** DreamLite conditions the UNet on Qwen3-VL's
`hidden_states[-1]`. In transformers 4.57.3, the version upstream pins,
that is the last decoder layer's output before the language model's final
norm; transformers 5 returns the tensor after that norm in the same slot.
Nothing fails, because the UNet's prompt projection ends in a norm of its
own, but the conditioning is a different tensor (about 70% relative
difference after the projection) and the output changes visibly: LPIPS
0.15 between the two versions' 4-step results at 640. The encode paths in
this repository (`pipeline_ops.prompt_hidden_state`) read the pre-norm
tensor on either version, after which the two agree to LPIPS 0.012.
Upstream's own `pipeline.__call__`, and the scripts here that call the
text encoder themselves (`champion_eval_token_pruning.py`,
`measure_te_batch_scaling.py`), still follow the installed version; run
those with transformers 4.57.3.

## Citation

Please cite the arXiv preprint:

```bibtex
@misc{ootani2026videoratestreamingstylizationvisionaware,
  title         = {Video-Rate Streaming Stylization on a Vision-Aware
                   MLLM-Conditioned Edit Diffusion: Asymmetric Batched
                   Inference on a Distilled UNet + MLLM Text Encoder},
  author        = {Yoshiyuki Ootani},
  year          = {2026},
  eprint        = {2606.05981},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.05981}
}
```

If you specifically want to reference this code release rather than the
paper, also cite the archived software deposit:

```bibtex
@software{ootani2026dreamlite_stream,
  author  = {Ootani, Yoshiyuki},
  title   = {dreamlite-stream: Video-Rate Streaming Stylization on a
             Vision-Aware MLLM-Conditioned Edit Diffusion},
  year    = {2026},
  version = {v0.1.0-preprint},
  doi     = {10.5281/zenodo.20389428},
  url     = {https://github.com/otanl/dreamlite-stream},
  note    = {Companion code to arXiv:2606.05981}
}
```

## License

Dual-licensed:

- **Code** — Apache License 2.0 — see [`LICENSE`](LICENSE).
- **Trained Temporal LLLite weights** under `runs/temporal_lllite_v3/`
  — inherit DreamLite's CC BY-NC 4.0 weight license, because they
  are Adapted Material of DreamLite-mobile (CC BY-NC 4.0 §1(a)).
  See [`ATTRIBUTION.md`](ATTRIBUTION.md) for the full attribution
  chain and downstream redistribution requirements.

The DreamLite-mobile UNet and Qwen3-VL TE are not part of this
repository and are subject to their upstream licenses.
