#!/bin/bash
# Setup DreamLite-stream env on vast.ai RTX 5090 (Blackwell, SM_120)
# Requires torch 2.7+ for native Blackwell compute capability support.
set -e
cd /workspace

# Show GPU + driver
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
echo "---"

[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet

# torch 2.7+ for Blackwell SM_120; try cu126 first, fallback cu124
echo "[install] torch (Blackwell-compatible)"
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126 --quiet \
  || pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet \
  || pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu126 --quiet

# Verify CUDA + Blackwell capability
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
print('compute_cap:', torch.cuda.get_device_capability(0))
"

# Core deps (versions exist for python 3.12)
echo "[install] core deps"
pip install \
  'transformers==4.57.3' \
  'diffusers==0.37.1' \
  accelerate safetensors einops tqdm \
  'opencv-python-headless==4.10.0.84' \
  pillow numpy imageio imageio-ffmpeg \
  --quiet

# xformers — may not have Blackwell wheel yet; allow skip
pip install xformers --quiet || echo "xformers skipped (no Blackwell wheel)"

echo ""
echo "[verify]"
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
import transformers, diffusers, cv2
print('hf', transformers.__version__, 'diffusers', diffusers.__version__, 'cv2', cv2.__version__)
# Test a minimal compile / matmul
x = torch.randn(8, 4096, 2048, device='cuda', dtype=torch.bfloat16)
y = torch.randn(8, 2048, 2048, device='cuda', dtype=torch.bfloat16)
z = torch.matmul(x, y)
print('matmul OK, output shape', z.shape, 'dtype', z.dtype)
"
echo "[setup done]"
