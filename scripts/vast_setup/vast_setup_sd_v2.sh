#!/bin/bash
# SD vid2vid benchmark setup — uses pre-patched local SD source
set -e
cd /workspace/StreamDiffusion
[ -d sd-venv ] || python3 -m venv sd-venv
source sd-venv/bin/activate
pip install --upgrade pip --quiet

# torch matching CUDA 12.4 driver
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet

# Versions compatible with our patches in src/streamdiffusion/acceleration/tensorrt/
pip install \
  "diffusers==0.30.3" \
  "transformers==4.40.0" \
  accelerate \
  "huggingface_hub<0.25" \
  peft fire omegaconf colored \
  --quiet

# ONNX + TRT (newer, matching our utilities.py rewrite for TRT 10 API)
pip install \
  "onnx==1.17.0" \
  "onnxruntime==1.18.1" \
  "cuda-python<13" \
  onnx-graphsurgeon \
  "polygraphy==0.49.14" \
  --quiet --extra-index-url https://pypi.ngc.nvidia.com

# TensorRT 10.6 cu12 — matches our local working setup
pip install "tensorrt-cu12==10.6.0.post1" --quiet

# Install streamdiffusion from local patched source
pip install -e . --no-deps --quiet

echo "[verify]"
python -c "
import torch, tensorrt, diffusers, transformers
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('trt', tensorrt.__version__, 'diffusers', diffusers.__version__, 'hf', transformers.__version__)
from streamdiffusion.acceleration.tensorrt.utilities import Engine
print('Engine import OK')
"
echo "[setup done]"
