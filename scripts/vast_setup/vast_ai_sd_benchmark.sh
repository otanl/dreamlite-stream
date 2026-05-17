#!/bin/bash
# StreamDiffusion benchmark on vast.ai 4090 — Phase 1
# Goal: reproduce kohaku-v2.1 + LCM-LoRA + TensorRT, K=1, 512x512, 100 iter
# Expected output: Average FPS line that matches StreamDiffusion's reported ~38 fps Kohaku K=4 region
# Time: ~15-20 min including TRT engine build

set -e
echo "[setup] $(nvidia-smi --query-gpu=name --format=csv,noheader) on $(uname -a)"

cd ~
[ -d StreamDiffusion ] || git clone https://github.com/cumulo-autumn/StreamDiffusion.git
cd StreamDiffusion

# Use system python or vast.ai's pre-installed pytorch env
PYTHON=${PYTHON:-python3}
$PYTHON -m venv sd-venv
source sd-venv/bin/activate
pip install --upgrade pip

# Install torch matching vast.ai's CUDA version
CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "[cuda] driver: $CUDA_VER"
# Try cu124 (modern), then cu121, then plain
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124 \
  || pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu121 \
  || pip install torch torchvision

# Core deps — pin to versions known to work with SD's TRT path
pip install diffusers==0.24.0 transformers==4.40.0 accelerate xformers \
  fire omegaconf colored "huggingface_hub<0.25" peft \
  onnx==1.15.0 onnxruntime==1.16.3 "cuda-python<12" \
  onnx-graphsurgeon polygraphy \
  --extra-index-url https://pypi.ngc.nvidia.com

# TensorRT 8.6 (Linux) — closest to what SD was written against
pip install "tensorrt==8.6.1.6" || pip install tensorrt

# Install streamdiffusion from source
pip install -e . --no-deps

echo "[verify] TRT import"
$PYTHON -c "import tensorrt, onnx, onnxruntime, onnx_graphsurgeon; \
  print('TRT', tensorrt.__version__); \
  from cuda import cudart; print('cudart OK')"

# Run benchmark — 100 iterations, K=1 LCM-LoRA, 512x512
cd examples/benchmark
echo "[bench] running single.py with --acceleration tensorrt"
$PYTHON -u single.py \
  --iterations 100 --warmup 10 \
  --width 512 --height 512 \
  --acceleration tensorrt \
  2>&1 | tee /tmp/sd_4090_trt.log

echo "[done]"
grep -E "Average (FPS|time)|Max FPS|Min FPS|Std" /tmp/sd_4090_trt.log
