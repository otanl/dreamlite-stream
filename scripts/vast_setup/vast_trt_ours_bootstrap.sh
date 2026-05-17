#!/bin/bash
# DreamLite UNet TRT bootstrap on vast.ai 4090
# Uses /workspace/venv (our dreamlite-stream env from Phase 2)
set -e
cd /workspace

source venv/bin/activate
echo "[env] $(python --version)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 1. Add TRT deps to our env (compatible — only new packages, no conflicts)
echo "[install] tensorrt + onnx tools"
pip install --quiet \
  "tensorrt-cu12==10.6.0.post1" \
  "onnx==1.17.0" \
  "onnxruntime==1.18.1" \
  "cuda-python<13" \
  onnx-graphsurgeon \
  "polygraphy==0.49.14" \
  --extra-index-url https://pypi.ngc.nvidia.com

python -c "
import tensorrt, onnx, onnxruntime
print('trt', tensorrt.__version__, 'onnx', onnx.__version__, 'ort', onnxruntime.__version__)
"

# 2. Export DreamLite UNet to ONNX (B=8, 512x512)
mkdir -p /workspace/dreamlite-stream/out/trt
cd /workspace/dreamlite-stream

echo ""
echo "[export] DreamLite UNet -> ONNX"
python scripts/export_unet_onnx.py \
  --out /workspace/dreamlite-stream/out/trt/unet_b8_512.onnx \
  --batch_size 8 --size 512 --prompt_seq_len 200 --dtype fp16 \
  2>&1 | tail -15

ls -lah /workspace/dreamlite-stream/out/trt/unet_b8_512.onnx

# 3. Build FP16 TRT engine
echo ""
echo "[build] TRT engine FP16"
python scripts/build_trt_engine.py \
  --onnx /workspace/dreamlite-stream/out/trt/unet_b8_512.onnx \
  --engine /workspace/dreamlite-stream/out/trt/unet_b8_512_fp16.engine \
  --workspace_gb 12 --fp16 \
  2>&1 | tail -10

ls -lah /workspace/dreamlite-stream/out/trt/unet_b8_512_fp16.engine

# 4. Numerical check (cosine sim vs PyTorch reference)
echo ""
echo "[validate] numerical accuracy"
python scripts/test_trt_unet.py \
  --engine /workspace/dreamlite-stream/out/trt/unet_b8_512_fp16.engine \
  --batch_size 8 --size 512 --prompt_seq_len 200 \
  --n_runs 10 --n_warmup 3 \
  2>&1 | tail -20

# 5. Head-to-head benchmark
echo ""
echo "[bench] TRT vs torch.compile (no LLLite, both)"
python scripts/bench_trt_vs_compile.py \
  --engine /workspace/dreamlite-stream/out/trt/unet_b8_512_fp16.engine \
  --video assets/davis_mp4/dance-twirl.mp4 \
  --batch_size 8 --size 512 \
  --n_batches 16 --n_warmup 2 \
  --mode both \
  2>&1 | tee /root/trt_ours_bench.log

echo "[done]"
