"""Optical-flow utilities for temporal latent warping.

Pipeline:
  cv2 Farneback flow on the **input** frames (grayscale, image-space)
  -> resize flow to latent resolution and divide displacement by vae_scale_factor
  -> torch.nn.functional.grid_sample to warp the **previous output latent**
     into the current frame's coordinate system

Design rationale:
  Computing flow on input frames (not on outputs) avoids feeding the
  stylization into the motion estimate — the assumption is that the
  underlying scene motion is the same in input and output. Warping the
  previous OUTPUT latent (not the previous output image) keeps the operation
  in the latent space the diffusion model already operates in.

cv2 Farneback was chosen over RAFT because it's CPU, dependency-free, and
"good enough" for relative comparisons. Cost ~5-10ms per 512x512 pair.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def farneback_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Image-space dense flow (H, W, 2). Float32 displacements (dx, dy)."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def warp_latent(
    latent: torch.Tensor,
    flow_img: np.ndarray,
    vae_scale_factor: int,
) -> torch.Tensor:
    """Forward-warp `latent` from time t to time t+1 using flow F: t -> t+1.

    For each output position (x', y') at time t+1, we sample the latent at
    its source position in time t, which (under the smooth-flow approximation)
    is (x' - F.x, y' - F.y). I.e., **gather with MINUS flow**, not plus.

    Common confusion: the "warping_error" metric uses cv2.remap with PLUS flow
    because it warps backward (out_{t+1} -> out_t). For temporal speculation
    we want forward warp (out_t -> out_{t+1}), so the sign is opposite.
    """
    B, C, h, w = latent.shape
    H, W = flow_img.shape[:2]
    # Resize flow to latent resolution and rescale displacement magnitudes.
    flow_lat = cv2.resize(flow_img, (w, h), interpolation=cv2.INTER_LINEAR)
    flow_lat[..., 0] *= w / W
    flow_lat[..., 1] *= h / H

    # Forward warp via gather: src_xy = output_xy - flow_at_output_xy
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    src_x = xs - flow_lat[..., 0]
    src_y = ys - flow_lat[..., 1]
    norm_x = (src_x / max(w - 1, 1)) * 2.0 - 1.0
    norm_y = (src_y / max(h - 1, 1)) * 2.0 - 1.0
    grid = np.stack([norm_x, norm_y], axis=-1)[None]  # (1, h, w, 2)
    grid_t = torch.from_numpy(grid).to(device=latent.device, dtype=latent.dtype)
    if B > 1:
        grid_t = grid_t.expand(B, -1, -1, -1)
    return F.grid_sample(
        latent, grid_t, mode="bilinear", padding_mode="border", align_corners=True,
    )


def to_gray(rgb_uint8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
