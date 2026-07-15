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
#   out/champion/results.jsonl   (per-sequence fps + warp_err)
```

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

# Recommended config for a smooth live feed (TE caching + LLLite cond
# refresh + fixed denoise noise; details in scripts/demo_camera.py):
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
pipeline, one frame paced on the display side). Step-only throughput
matches the paper's per-batch sustained number; the demo's wall-clock
fps is slightly lower because it includes display pacing and frame-
drop bookkeeping.

**RIFE setup** (only needed if you pass `--interp_method rife`):

```bash
# Sibling clone of Practical-RIFE (MIT-licensed)
git clone https://github.com/hzwer/Practical-RIFE ../Practical-RIFE
# Download a RIFE_HDv3 checkpoint from the link in their README
# and place it under ../Practical-RIFE/train_log/

# Then launch the demo with:
python scripts/demo_camera.py --prompt "..." \
    --interp_factor 2 --interp_method rife \
    --rife_path ../Practical-RIFE
```

Useful tuning flags (`--help` for the full list):

| Flag | What it does |
|---|---|
| `--te_refresh_every N` | Cache Q3-VL prompt_embeds across N batches. Hides the ~250 ms TE-forward cost on the cached majority of iters; trades static-prompt image-awareness for throughput. |
| `--te_batch_one` | When refreshing, run Q3-VL on a single representative frame and broadcast — multimodal sequence shortens ~B×. Only sensible when paired with `--te_refresh_every > 1`. |
| `--cond_refresh_every N` | Same amortisation for the LLLite temporal cond (Farneback flow + warp). Lower N → better temporal coherence on live camera; higher N → faster throughput. |
| `--fixed_noise` | Reuse the same denoise init noise pattern across frames — reduces stochastic flicker; the paper's measurements all use this. |
| `--quant_te {nf4,fp4,int8}` | Quantise Q3-VL on load (requires `bitsandbytes`). For fitting in lower VRAM; speed-neutral on ≥12 GB hosts. |
| `--compile_mode reduce-overhead` | Best demo throughput; enables `torch.compile`'s CUDA Graph capture (requires `triton`). |
| `--interp_factor N` | Display-side frame interpolation. `2` doubles, `4` quadruples the displayed fps without invoking the model again. Method is chosen by `--interp_method`. Pipeline throughput unchanged. |
| `--interp_method {linear,rife}` | `linear` (default) is a cheap pixel blend, mild ghosting on fast motion. `rife` calls [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) per pair on the output side (input side always uses linear). RIFE adds ~5–15 ms per pair but recovers fast-motion fidelity. Requires `--rife_path` (and optionally `--rife_model`). |
| `--temporal_blend_alpha F` | Sequential temporal blend on output frames (mix `1-α` of current with `α` of predecessor). Reduces static-region flicker; over-strong values blur fast motion. `0.0` = off (default), `0.3–0.6` is the useful range. |
| `--temporal_blend_warp` | When blending, Farneback-warp the predecessor toward the current frame first. Removes most motion ghosting at the cost of ~5 ms CPU per frame. Pair with `--temporal_blend_alpha > 0`. |

## Reproducing paper tables and figures

Table labels below match the arXiv preprint
(Roman-numeral table numbering; content described in case of future
re-numbering).

| Paper artefact (content) | Script | Notes |
|---|---|---|
| Table II — Main DAVIS-2017 results | `scripts/comprehensive_ablation.py` | all rows of the speed pipeline + LLLite + hook subset |
| Table III — Sustained throughput + latency | `scripts/sustained_throughput_test.py` | 480-frame *parkour* loop, p50/p95 latency |
| Table IV — In-pipeline component profile | `scripts/profile_inpipeline.py` | TE/UNet/VAE per-batch wall, side vs. main |
| Table V — Per-component batch scaling | `scripts/pareto_batch_sweep.py` | sub-linearity per component |
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
| Negative result — TensorRT path | `scripts/export_unet_lllite_onnx.py`, `scripts/build_trt_engine_mixed.py`, `scripts/debug_lllite_trt_drift.py` | LLLite-baked TRT export attempt |

Pre-computed per-row results are committed under `out/.../results.jsonl`
where the file size allows, so reviewers can verify the table numbers
without rerunning. (Mp4 outputs themselves are not committed; see
`.gitignore`.)

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
- **Per-sequence JSONL** evaluation logs are committed under
  `out/champion/results.jsonl`,
  `out/comprehensive_ablation/results.jsonl`, etc.
- **Synthetic scene-cut clips** (Appendix B) under `assets/scene_cut/`.

## Reproducibility caveats

We do not redistribute the DreamLite-mobile UNet weights or the
Qwen3-VL TE weights. The preprint's Reproducibility section documents
what can and cannot be verified without upstream weight access:
the **inference pipeline reformulation**, the
**compile-friendly LLLite module**, the **trained adapter**, the
**benchmark scripts**, the **TRT export tooling**, and the
**per-sequence JSONL** profiler logs are all in this repo. The
end-to-end headline numbers require obtaining the upstream gated
weights under their respective licenses.

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
