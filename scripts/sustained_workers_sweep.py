"""Sweep cond_flow_workers to find best threading config."""
from __future__ import annotations
import sys, time, warnings
from itertools import cycle, islice
from pathlib import Path
from statistics import mean, stdev
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))
warnings.filterwarnings("ignore")
from PIL import Image
import cv2
from dreamlite import DreamLiteMobilePipeline
from dreamlite_lllite.inject import apply_lllite
from safetensors.torch import load_file
from dreamlite_stream import BatchedEditWorker, SharedState

print("[load] pipeline")
pipeline = DreamLiteMobilePipeline.from_pretrained(
    str(_DREAMLITE / "models" / "DreamLite-mobile"), torch_dtype=torch.bfloat16
).to("cuda")
sd = load_file(str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
vae_d = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
controller = apply_lllite(
    pipeline.unet, cond_emb_dim=32, mlp_dim=64,
    cond_image_size=512, sample_size=512 // vae_d,
    block_filter=["down_blocks"], inference_mode=True, max_batch_size=16,
)
controller.load_state_dict(sd, strict=False)
controller.to(device="cuda", dtype=torch.bfloat16)
controller.eval()
controller.set_multiplier(1.0)

cap = cv2.VideoCapture(str(_ROOT / "assets" / "davis_mp4" / "parkour.mp4"))
raw = []
while True:
    ok, f = cap.read()
    if not ok: break
    f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    f_rgb = cv2.resize(f_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    raw.append(Image.fromarray(f_rgb))
cap.release()
frames = list(islice(cycle(raw), 320))

print(f"\n{'workers':>8} {'mean fps':>10} {'p50 ms':>8} {'p95 ms':>8} {'std ms':>8}")
for nw in [1, 4, 8, 16, 32]:
    state = SharedState(height=512, width=512, num_inference_steps=1,
                       prompt="transfer this to oil painting style, vibrant colors")
    worker = BatchedEditWorker(
        pipeline=pipeline, state=state, batch_size=16, device="cuda",
        dtype=torch.bfloat16, seed=42, compile=True, compile_mode="reduce-overhead",
        lllite_controller=controller, cond_refresh_every=8,
        cond_flow_workers=nw,
    )
    walls = []
    cur_buf = cur_pf = None
    for b in range(20):
        buf = frames[b*16:(b+1)*16]
        if cur_buf is None:
            cur_buf = buf; cur_pf = worker.prefetch_batch(cur_buf); continue
        nxt = worker.prefetch_batch(buf)
        t0 = time.perf_counter()
        worker.step_batch_with_prefetch(cur_buf, cur_pf)
        torch.cuda.synchronize()
        if b >= 4:
            walls.append((time.perf_counter() - t0) * 1000)
        cur_buf, cur_pf = buf, nxt
    n = len(walls)
    s = sorted(walls)
    p50, p95 = s[n//2], s[int(n*0.95)]
    fps = (n * 16) / (sum(walls) / 1000)
    print(f"{nw:>8} {fps:>10.2f} {p50:>8.1f} {p95:>8.1f} {stdev(walls) if n>=2 else 0:>8.1f}")
