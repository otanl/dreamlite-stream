# dreamlite-stream

Video-rate streaming stylization on the smallest vision-aware
MLLM-conditioned edit-diffusion stack.

> ⚠️ **Non-commercial, research-only.** The trained adapter
> weights in this repository inherit DreamLite-mobile's CC BY-NC 4.0
> weight licence and the upstream use policy
> (no NSFW / violent / discriminatory / illegal content, no
> commercial use, no malicious use). See [`USE_POLICY.md`](USE_POLICY.md)
> for the verbatim upstream notice and what it means in practice.

This repository accompanies the TMLR submission **_Video-Rate
Streaming Stylization on the Smallest Vision-Aware MLLM-Conditioned
Edit Diffusion_** and contains the inference runtime, training
scripts, evaluation harness, and a trained Temporal LLLite adapter
(v3) used to produce the paper's tables and figures.

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
sequences from Table 4 of the paper:

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

Useful tuning flags (`--help` for the full list):

| Flag | What it does |
|---|---|
| `--te_refresh_every N` | Cache Q3-VL prompt_embeds across N batches. Hides the ~250 ms TE-forward cost on the cached majority of iters; trades static-prompt image-awareness for throughput. |
| `--te_batch_one` | When refreshing, run Q3-VL on a single representative frame and broadcast — multimodal sequence shortens ~B×. Only sensible when paired with `--te_refresh_every > 1`. |
| `--cond_refresh_every N` | Same amortisation for the LLLite temporal cond (Farneback flow + warp). Lower N → better temporal coherence on live camera; higher N → faster throughput. |
| `--fixed_noise` | Reuse the same denoise init noise pattern across frames — reduces stochastic flicker; the paper's measurements all use this. |
| `--quant_te {nf4,fp4,int8}` | Quantise Q3-VL on load (requires `bitsandbytes`). For fitting in lower VRAM; speed-neutral on ≥12 GB hosts. |
| `--compile_mode reduce-overhead` | Best demo throughput; enables `torch.compile`'s CUDA Graph capture (requires `triton`). |

## Reproducing paper tables and figures

The numbered references match the submitted manuscript.

| Paper artefact | Script | Notes |
|---|---|---|
| Table 4 (main ablation) | `scripts/comprehensive_ablation.py` | all rows of the speed pipeline + LLLite + hook subset |
| Table 6 (in-pipeline profile) | `scripts/profile_inpipeline.py` | TE/UNet/VAE per-batch wall, side vs. main |
| Table 8 (cond-refresh sweep) | `scripts/cond_refresh_sweep_downblocks.py` + `scripts/cond_refresh_spatial_metrics.py` | N in {1,4,8,16} fps/εw plus Sobel/HF-FFT/LPIPS-to-N=1 |
| Table 10 (3-video held-out) | `scripts/eval_heldout_video.py` | requires the v5 held-out checkpoint |
| Table 11 (held-out prompts) | `scripts/eval_heldout_prompts.py` | comic / ukiyo-e / van Gogh |
| Table 12 (multi-prompt v4) | `scripts/eval_multiprompt.py` | 5-prompt v4 LLLite |
| Table 13 (cross-dataset) | `scripts/phase3_crossds_local.py` | 15 clips from 7 non-DAVIS sources |
| Table 14 (DAVIS-19 unused held-out) | `scripts/champion_eval.py` with `--sequences` overridden | see paper §heldout |
| Table 15 (sustained throughput) | `scripts/sustained_throughput_test.py` | 480-frame run + latency p50/p95 |
| Tables 17, 18 (RAFT cross-check, Appendix A) | `scripts/raft_warp_error.py`, `scripts/raft_warp_error_ordering.py` | Farneback vs. RAFT |
| Tables 19, 20 (scene-cut, Appendix B) | `scripts/scene_cut_eval.py`, `scripts/scene_cut_lpips.py` | synthetic hard cuts |
| Figure 1 (smoothing artifact) | `scripts/viz_temporal_pair.py` | LCM-LoRA vs. LLLite vs. baseline |
| Figure 2 (pipeline schematic) | — | drawn from Table 6 numbers, source SVG in the paper repo |
| Figures 3, 4 (qualitative grids) | `scripts/extract_figure_frames.py` + `build_qualitative_grid_svg.py` + `build_scenecut_grid_svg.py` | regenerate from the saved mp4s |
| Negative result 5 (token pruning) | `scripts/champion_eval_token_pruning.py` | pruned-prompt eval |
| TensorRT path (negative result 4) | `scripts/export_unet_lllite_onnx.py`, `scripts/build_trt_engine_mixed.py`, `scripts/debug_lllite_trt_drift.py` | LLLite-baked TRT export attempt |

Pre-computed per-row results are committed under `out/.../results.jsonl`
where the file size allows, so reviewers can verify the table numbers
without rerunning. (Mp4 outputs themselves are not committed; see
`.gitignore`.)

## Released artefacts

- **v3 Temporal LLLite checkpoint** (~51 MB; 38 attention hooks on
  the DreamLite-mobile UNet `down_blocks` after load-time subset
  filtering) — **not committed to this repository.** The weights
  are Adapted Material of DreamLite-mobile and inherit
  DreamLite's CC BY-NC 4.0 weight licence; pending the upstream
  attribution check we host them externally (HuggingFace release
  link to be added here once available). The training-args
  metadata is committed under `runs/temporal_lllite_v3/args.json`
  so the recipe is fully reproducible from the training script
  (`scripts/train_temporal_lllite.py`) given the upstream
  DreamLite-mobile weights.
- **Per-sequence JSONL** evaluation logs are committed under
  `out/champion/results.jsonl`,
  `out/comprehensive_ablation/results.jsonl`, etc.
- **Synthetic scene-cut clips** (Appendix B) under `assets/scene_cut/`.

## Reproducibility caveats

We do not redistribute the DreamLite-mobile UNet weights or the
Qwen3-VL TE weights. The TMLR submission's Reproducibility section
documents what can and cannot be verified without upstream weight
access: the **inference pipeline reformulation**, the
**compile-friendly LLLite module**, the **trained adapter**, the
**benchmark scripts**, the **TRT export tooling**, and the
**per-sequence JSONL** profiler logs are all in this repo. The
end-to-end headline numbers require obtaining the upstream gated
weights under their respective licenses.

## Citation

```bibtex
@article{anonymous2026dreamlitestream,
  title   = {Video-Rate Streaming Stylization on the Smallest
             Vision-Aware MLLM-Conditioned Edit Diffusion:
             Periodic Cond-Refresh and Asymmetric Batched Inference
             on Distilled UNet + MLLM TE},
  author  = {Anonymous},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  note    = {Under review}
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
