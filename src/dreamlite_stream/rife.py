"""Batched RIFE frame interpolation for the display side.

The pipeline emits one stylized frame per source frame. `RIFEInterpolator`
inserts flow-based in-between frames so the display runs at a multiple of
the pipeline rate. One call handles a whole batch: the N pairs
(previous frame, frame 0), (frame 0, frame 1), ... go through the network
together, and the last frame is carried over so the pair across a batch
boundary is interpolated too.

The network is RIFE v4.25 / v4.26 (IFNet, inference path only), re-written
from IFNet_HDv3.py and warplayer.py of Practical-RIFE
(https://github.com/hzwer/Practical-RIFE, Copyright (c) 2021 hzwer, MIT
License; the notice is reproduced in ATTRIBUTION.md). Only its checkpoint
(`flownet.pkl`) is needed; the module definitions below match the
checkpoint's parameter names. Differences from the reference forward, none
of which change the result beyond rounding:

* every frame is encoded once, although it takes part in two pairs;
* the sampling grid of the warps is computed in fp32 while the convolutions
  run in fp16 (fp16 cannot place a sample to sub-pixel accuracy at these
  sizes);
* the time position is a tensor input, so one TensorRT engine serves every
  in-between position.

Two backends: PyTorch (fp16 / channels_last), and a TensorRT engine built by
`scripts/build_trt_rife.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALES = (16, 8, 4, 2, 1)       # flow estimated coarse to fine
MULTIPLE = 64                   # frame sides must be a multiple of this
ENGINE_FORMAT = "dreamlite-stream/trt-rife/1"


def _conv(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.LeakyReLU(0.2, True))


class Head(nn.Module):
    """Per-frame feature encoder: 3 -> 4 channels at full resolution."""

    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """One refinement level: works at 1/scale resolution, returns full-resolution flow, mask, feature."""

    def __init__(self, in_planes: int, c: int):
        super().__init__()
        self.conv0 = nn.Sequential(_conv(in_planes, c // 2, 2), _conv(c // 2, c, 2))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow, scale: int):
        h, w = x.shape[-2:]
        small = (h // scale, w // scale)
        if scale != 1:
            x = F.interpolate(x, size=small, mode="bilinear", align_corners=False)
        if flow is not None:
            if scale != 1:
                flow = F.interpolate(flow, size=small, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        out = self.lastconv(self.convblock(self.conv0(x)))
        if scale != 1:
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out[:, :4] * scale, out[:, 4:5], out[:, 5:]


def _base_grid(h: int, w: int, device) -> torch.Tensor:
    """(1, 2, h, w) fp32 sampling coordinates of the identity warp, in grid_sample's [-1, 1] range."""
    xs = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, 1, w).expand(1, 1, h, w)
    ys = torch.linspace(-1.0, 1.0, h, device=device).view(1, 1, h, 1).expand(1, 1, h, w)
    return torch.cat([xs, ys], 1)


def _warp(x: torch.Tensor, flow: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Backward-warp `x` by `flow` (pixels). The coordinates are fp32 whatever the dtype of `x`."""
    h, w = x.shape[-2:]
    f = flow.float()
    g = grid + torch.cat([f[:, 0:1] / ((w - 1.0) / 2.0), f[:, 1:2] / ((h - 1.0) / 2.0)], 1)
    out = F.grid_sample(x.float(), g.permute(0, 2, 3, 1), mode="bilinear", padding_mode="border",
                        align_corners=True)
    return out.to(x.dtype)


class IFNet(nn.Module):
    """RIFE v4.25 flow network. `c4` is the width of the last block: 32, or 24 for the lite model."""

    def __init__(self, c4: int = 32):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=c4)
        self.encode = Head()

    def forward(self, frames: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """frames: (N+1, 3, H, W) in [0, 1]; t: one value in (0, 1). Returns the N in-between
        frames at position t between frames[i] and frames[i+1]."""
        n = frames.shape[0] - 1
        h, w = frames.shape[-2:]
        img0, img1 = frames[:-1], frames[1:]
        feat_all = self.encode(frames)
        f0, f1 = feat_all[:-1], feat_all[1:]
        tmap = t.to(frames.dtype).reshape(1, 1, 1, 1).expand(n, 1, h, w)
        grid = _base_grid(h, w, frames.device)
        flow = mask = feat = None
        w0, w1 = img0, img1
        for block, scale in zip((self.block0, self.block1, self.block2, self.block3, self.block4), SCALES):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, tmap), 1), None, scale)
            else:
                wf0 = _warp(f0, flow[:, :2], grid)
                wf1 = _warp(f1, flow[:, 2:4], grid)
                fd, mask, feat = block(torch.cat((w0, w1, wf0, wf1, tmap, mask, feat), 1), flow, scale)
                flow = flow + fd
            w0 = _warp(img0, flow[:, :2], grid)
            w1 = _warp(img1, flow[:, 2:4], grid)
        m = torch.sigmoid(mask)
        return w0 * m + w1 * (1 - m)


def load_ifnet(weights) -> IFNet:
    """Build the network for a Practical-RIFE `flownet.pkl` (v4.25, v4.25 lite or v4.26)."""
    path = Path(weights)
    if path.is_dir():
        path = path / "flownet.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download RIFE v4.25 from the links in "
            "https://github.com/hzwer/Practical-RIFE and pass its train_log/flownet.pkl")
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    try:
        c4 = state["block4.conv0.1.0.weight"].shape[0]
    except KeyError as e:
        raise ValueError(f"{path} is not a RIFE v4.25-style checkpoint (5 flow blocks expected)") from e
    net = IFNet(c4=c4)
    own = net.state_dict()
    missing = [k for k in own if k not in state]
    wrong = [k for k in own if k in state and state[k].shape != own[k].shape]
    if missing or wrong:
        raise ValueError(f"{path} does not match RIFE v4.25: {len(missing)} missing, {len(wrong)} mis-shaped "
                         f"parameters (first: {(missing + wrong)[0]})")
    net.load_state_dict({k: state[k] for k in own})
    return net.eval()


def engine_spec_path(engine_path) -> Path:
    return Path(engine_path).with_suffix(".json")


class _TRTRunner:
    """Runs a build_trt_rife.py engine: inputs `frames` (N+1, 3, H, W) and `t` (1,), output (N, 3, H, W)."""

    def __init__(self, engine_path, device):
        import tensorrt as trt

        spec_file = engine_spec_path(engine_path)
        if not spec_file.exists():
            raise FileNotFoundError(f"{spec_file} not found; build the engine with scripts/build_trt_rife.py")
        self.spec = json.loads(spec_file.read_text(encoding="utf-8"))
        if self.spec.get("format") != ENGINE_FORMAT:
            raise ValueError(f"{spec_file}: unknown engine spec format {self.spec.get('format')!r}")
        if self.spec.get("tensorrt") != trt.__version__:
            raise RuntimeError(f"{engine_path} was built with TensorRT {self.spec.get('tensorrt')}; this is "
                               f"{trt.__version__}. Rebuild it here.")
        if self.spec.get("platform") not in (None, "", sys.platform):
            raise RuntimeError(f"{engine_path} was built on {self.spec['platform']}; this is {sys.platform}. "
                               "Engines do not carry over between platforms; build one here.")
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self._runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not load {engine_path}")
        self.context = self.engine.create_execution_context()
        shape = tuple(int(d) for d in self.engine.get_tensor_shape("frames"))
        self.n, self.hw = shape[0] - 1, shape[2:]
        self.out = torch.empty((self.n, 3, *self.hw), dtype=torch.float16, device=device)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)  # TensorRT adds host syncs on the default stream

    def __call__(self, frames: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        frames = frames.to(torch.float16).contiguous()
        t = t.to(torch.float16).reshape(1).contiguous()
        self.context.set_tensor_address("frames", frames.data_ptr())
        self.context.set_tensor_address("t", t.data_ptr())
        self.context.set_tensor_address("mids", self.out.data_ptr())
        self.stream.wait_stream(torch.cuda.current_stream(self.device))  # the inputs are ready
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()  # frames / t are released on return
        return self.out


class RIFEInterpolator:
    """Inserts in-between frames into batches of uint8 frames.

    weights:  Practical-RIFE `flownet.pkl` (or its folder). Not needed with `engine`.
    engine:   a TensorRT engine from scripts/build_trt_rife.py; fixes the frame size and batch.
    """

    def __init__(self, weights=None, engine=None, device: str = "cuda"):
        self.device = torch.device(device)
        self._trt: Optional[_TRTRunner] = None
        self._net: Optional[IFNet] = None
        if engine:
            self._trt = _TRTRunner(engine, self.device)
        else:
            if weights is None:
                raise ValueError("pass RIFE weights or a TensorRT engine")
            self._net = load_ifnet(weights).to(self.device, torch.float16).to(memory_format=torch.channels_last)
        self._prev: Optional[torch.Tensor] = None   # last frame of the previous batch, (1, 3, H, W) fp16

    @property
    def backend(self) -> str:
        return "tensorrt" if self._trt is not None else "pytorch"

    def reset(self) -> None:
        """Forget the carried frame (call between unrelated clips)."""
        self._prev = None

    # ------------------------------------------------------------------
    def _run(self, frames: torch.Tensor, t: float) -> torch.Tensor:
        """frames (N+1, 3, H, W) fp16 in [0, 1] -> N in-between frames."""
        tt = torch.tensor([t], device=self.device, dtype=torch.float16)
        if self._trt is None:
            return self._net(frames.contiguous(memory_format=torch.channels_last), tt)
        n = frames.shape[0] - 1
        if n > self._trt.n:
            raise ValueError(f"{n} frames exceed the engine's batch size {self._trt.n}")
        if n < self._trt.n:  # pad with repeats of the last frame; their in-betweens are dropped
            frames = torch.cat([frames, frames[-1:].expand(self._trt.n - n, -1, -1, -1)], 0)
        return self._trt(frames, tt)[:n]

    @torch.no_grad()
    def expand(self, frames_u8, factor: int = 2, bgr: bool = False) -> np.ndarray:
        """Interleave in-between frames with a batch.

        frames_u8: (N, H, W, 3) uint8 array, or a list of N (H, W, 3) arrays.
        Returns (N * factor, H, W, 3) uint8: for each input frame, its
        factor-1 in-betweens from the frame before it, then the frame. The
        frame before the first one is the last frame of the previous call
        (on the first call, the first frame itself).
        """
        arr = frames_u8 if isinstance(frames_u8, np.ndarray) else np.stack(frames_u8)
        n, h, w, _ = arr.shape
        if factor < 2:
            return arr
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device).permute(0, 3, 1, 2)
        x = x.to(torch.float16).div_(255.0)
        if bgr:
            x = x.flip(1)
        ph, pw = (-h) % MULTIPLE, (-w) % MULTIPLE
        if self._trt is not None and (h + ph, w + pw) != tuple(self._trt.hw):
            raise ValueError(f"frames are {h}x{w}; the RIFE engine is built for {self._trt.hw}")
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        prev = self._prev if (self._prev is not None and self._prev.shape[-2:] == x.shape[-2:]) else x[:1]
        seq = torch.cat([prev, x], 0)
        self._prev = x[-1:].clone()

        mids = torch.empty((n, factor - 1, h, w, 3), dtype=torch.uint8, device=self.device)
        for k in range(1, factor):
            mid = self._run(seq, k / factor)[:, :, :h, :w]
            if bgr:
                mid = mid.flip(1)
            mids[:, k - 1] = mid.float().clamp_(0, 1).mul_(255.0).round_().to(torch.uint8).permute(0, 2, 3, 1)
        # Only the new frames come back from the GPU; the originals are copied as they are.
        out = np.empty((n, factor, h, w, 3), dtype=np.uint8)
        out[:, : factor - 1] = mids.cpu().numpy()
        out[:, factor - 1] = arr
        return out.reshape(n * factor, h, w, 3)


class LinearCarry:
    """The cross-fade counterpart of `RIFEInterpolator.expand`, for the camera side of a
    side-by-side view: same frame count and ordering, plain blends as in-betweens."""

    def __init__(self):
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None

    def expand(self, frames: Sequence[np.ndarray], factor: int = 2) -> List[np.ndarray]:
        import cv2

        out: List[np.ndarray] = []
        prev = self._prev
        for f in frames:
            if prev is None or prev.shape != f.shape:
                prev = f
            for k in range(1, factor):
                out.append(cv2.addWeighted(prev, 1.0 - k / factor, f, k / factor, 0))
            out.append(f)
            prev = f
        self._prev = prev
        return out
