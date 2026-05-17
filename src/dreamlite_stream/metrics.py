"""Temporal-consistency metrics for video edits.

We use simple, well-established metrics rather than learned ones, so the
implementation is dep-light (cv2 + numpy) and the numbers are inspectable.

  warping_error
    For each consecutive pair (in_t, in_{t+1}), compute optical flow on the
    INPUT (Farneback). Warp out_{t+1} back to t using that flow. Return the
    mean absolute photometric error over the sequence (pixel range 0-255).
    Lower = output respects the input motion more faithfully = less flicker.

  consecutive_l1
    Mean of |out_t - out_{t+1}| over the sequence. Includes both flicker AND
    legitimate motion, so this is a coarse number on its own.

  consistency_ratio
    warping_error / max(consecutive_l1, eps). 0 means all frame-to-frame
    change is explained by input motion (perfect temporal consistency); 1
    means none of it is. Useful for comparing configs on the same video.

References:
  Lai et al. 2018 "Learning Blind Video Temporal Consistency" use a more
  refined version with occlusion masks; we omit those for MVP since we are
  comparing configs run on the same input (so any occlusion error is shared).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def read_video_frames(path: str, size: Optional[int] = None) -> List[np.ndarray]:
    """Read all frames as RGB uint8 ndarrays. If `size` given, center-crop
    square then resize to (size, size) to match the runtime input pipeline."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if size is not None:
                h, w, _ = rgb.shape
                s = min(h, w)
                y0, x0 = (h - s) // 2, (w - s) // 2
                rgb = rgb[y0 : y0 + s, x0 : x0 + s]
                rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
    finally:
        cap.release()
    return frames


def _farneback_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def _warp_with_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xs + flow[..., 0]).astype(np.float32)
    map_y = (ys + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


@dataclass
class TemporalMetrics:
    warping_error: float          # mean |out_t - warp(out_{t+1}, flow_in)|
    consecutive_l1: float         # mean |out_t - out_{t+1}|
    consistency_ratio: float      # warping_error / consecutive_l1
    n_pairs: int


def compute_temporal(
    input_frames: List[np.ndarray], output_frames: List[np.ndarray],
) -> TemporalMetrics:
    if len(input_frames) != len(output_frames):
        raise ValueError(
            f"frame count mismatch: in={len(input_frames)} out={len(output_frames)}"
        )
    if len(input_frames) < 2:
        return TemporalMetrics(0.0, 0.0, 0.0, 0)

    we_sum = 0.0
    cl1_sum = 0.0
    n = 0
    prev_in_gray = cv2.cvtColor(input_frames[0], cv2.COLOR_RGB2GRAY)
    for t in range(len(input_frames) - 1):
        curr_in_gray = cv2.cvtColor(input_frames[t + 1], cv2.COLOR_RGB2GRAY)
        flow = _farneback_flow(prev_in_gray, curr_in_gray)

        out_a = output_frames[t].astype(np.float32)
        out_b = output_frames[t + 1].astype(np.float32)
        warped_b = _warp_with_flow(out_b, flow)

        we_sum += float(np.mean(np.abs(out_a - warped_b)))
        cl1_sum += float(np.mean(np.abs(out_a - out_b)))
        n += 1
        prev_in_gray = curr_in_gray

    we = we_sum / n
    cl1 = cl1_sum / n
    ratio = we / cl1 if cl1 > 1e-6 else 0.0
    return TemporalMetrics(
        warping_error=we, consecutive_l1=cl1, consistency_ratio=ratio, n_pairs=n,
    )


def reference_l1(
    test_frames: List[np.ndarray], ref_frames: List[np.ndarray],
) -> float:
    """Mean L1 between two frame sequences (uint8 RGB; pixel range 0-255).

    Honest fidelity metric for outputs that may use temporal warping internally:
    unlike `warping_error`, this cannot be gamed by constructing the output as
    a flow-warp of the previous output, because the reference is independent
    per-frame full-quality denoise.
    """
    n = min(len(test_frames), len(ref_frames))
    if n == 0:
        return 0.0
    return float(np.mean([
        np.mean(np.abs(test_frames[i].astype(np.float32) - ref_frames[i].astype(np.float32)))
        for i in range(n)
    ]))


def hf_density(frames: List[np.ndarray]) -> Tuple[float, float]:
    """Spatial high-frequency probe used as a style-fidelity floor.

    Returns (sobel_mean, hf_fft_mean):
      - sobel_mean: mean |Sobel(grayscale)| averaged over all frames.
      - hf_fft_mean: mean magnitude of FFT outside center radius H/4.

    Both rise with sharp brushstrokes / fine texture, fall with blur. Pair
    with warping_error to detect outputs that game warp_err by smoothing.
    """
    if not frames:
        return 0.0, 0.0
    sobel_acc = 0.0
    hf_acc = 0.0
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f
        sobel_acc += float(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)).mean())
        F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        H, W = gray.shape
        cy, cx = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]
        mask = ((Y - cy) ** 2 + (X - cx) ** 2) > (min(H, W) // 4) ** 2
        hf_acc += float(np.abs(F)[mask].mean())
    n = len(frames)
    return sobel_acc / n, hf_acc / n


def make_grid(
    frames_per_row: List[Tuple[str, np.ndarray]],
    label_height: int = 30,
) -> np.ndarray:
    """Lay out (label, frame) pairs left-to-right with a labelled bar above."""
    h, w, _ = frames_per_row[0][1].shape
    n = len(frames_per_row)
    canvas = np.full((h + label_height, w * n, 3), 32, dtype=np.uint8)
    for i, (label, img) in enumerate(frames_per_row):
        canvas[label_height : label_height + h, i * w : (i + 1) * w] = img
        cv2.putText(
            canvas, label, (i * w + 8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA,
        )
    return canvas
