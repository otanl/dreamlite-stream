# scripts/dev — historical / internal scripts

These are dev-history scripts that were used during development but are
**not part of the paper's reproduction surface**. They live here so the
top-level `scripts/` directory only contains things a reader of the
paper or repo actually needs.

If you're trying to reproduce a paper table or figure, you want
`scripts/<top-level>.py` — see the table in the top-level `README.md`.

## What's here and why

### MVP iteration history (superseded)
- `mvp1_video.py`, `mvp2_video.py`, `mvp2_benchmark.py`,
  `mvp3_batched.py` — early implementations of the inference loop.
  Superseded by `scripts/champion_eval.py` and the BatchedEditWorker
  in `src/dreamlite_stream/workers/batched_edit.py`.

### Early ablation / comparison runners (superseded)
- `ablation_main.py`, `ablation_summary.py` — superseded by
  `scripts/comprehensive_ablation.py`.
- `benchmark.py` — superseded by `scripts/sustained_throughput_test.py`.
- `final_compare.py`, `quality_compare.py`, `reference_compare.py` —
  earlier comparison utilities; the per-table comparison logic lives
  inside the individual table scripts now.

### Debugging / profiling scratch
- `drift_test.py`, `debug_trt_drift.py` — debug scripts used while
  tracking down the LLLite-baked TRT FP32 vs FP16 drift (the final
  understanding is documented in `scripts/debug_lllite_trt_drift.py`,
  which is the one we kept).
- `profile_prefetch.py`, `profile_champion.py`,
  `profile_cond_refresh.py` — interactive profiling sessions used to
  derive the per-stage numbers that ended up in Table 6
  (`scripts/profile_inpipeline.py`).

### Alternate TRT / ONNX exports (superseded)
- `bench_trt_vs_compile.py` — early A/B benchmark; the paper's TRT
  path is the LLLite-baked one in
  `scripts/build_trt_engine_mixed.py`.
- `build_trt_engine.py`, `export_unet_onnx.py` — non-LLLite-baked
  variants. Kept for archival but superseded by the `*_lllite_*`
  versions referenced in the paper's Negative Result 4.

### Design docs (internal)
- `cross_frame_attention_design.md`, `stream_batch_design.md` —
  exploratory design notes from before the final architecture
  settled. The final design lives in the source docstrings and
  `notes/demo_tuning.md` (gitignored).
