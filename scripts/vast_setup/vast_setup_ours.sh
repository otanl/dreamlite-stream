#!/bin/bash
# Setup DreamLite-stream env on vast.ai 4090 — Phase 2
set -e
cd /workspace
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet

# torch matching CUDA 12.4 driver
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet

# Core deps from dreamlite-lllite pyproject (most restrictive)
pip install \
  "transformers==4.57.3" \
  "diffusers==0.37.2" \
  accelerate safetensors einops tqdm \
  "opencv-python-headless==4.10.0.84" \
  pillow numpy imageio imageio-ffmpeg \
  --quiet

# xformers for memory efficiency (matching torch 2.6)
pip install "xformers==0.0.29.post2" --quiet || echo "xformers skipped"

# Verify
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import transformers, diffusers; print('hf', transformers.__version__, 'diffusers', diffusers.__version__)"
python -c "import cv2; print('cv2', cv2.__version__)"
echo "[setup done]"
