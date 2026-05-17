#!/bin/bash
# Setup DreamLite-stream env on vast.ai 4090 — Phase 2 (corrected versions)
set -e
cd /workspace
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet

# torch matching CUDA 12.4 driver
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet

# Core deps (versions exist for python 3.12)
pip install \
  "transformers==4.57.3" \
  "diffusers==0.37.1" \
  accelerate safetensors einops tqdm \
  "opencv-python-headless==4.10.0.84" \
  pillow numpy imageio imageio-ffmpeg \
  --quiet

# xformers matching torch 2.6
pip install "xformers==0.0.29.post2" --quiet || echo "xformers skipped"

echo "[verify]"
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
import transformers, diffusers, cv2
print('hf', transformers.__version__, 'diffusers', diffusers.__version__, 'cv2', cv2.__version__)
"
echo "[setup done]"
