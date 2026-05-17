"""Streaming runtime: video file → per-frame edit → output video + timing log."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


from .output_blend import OutputBlender
from .workers.edit import EditWorker, StepTiming


def _center_square_crop(rgb: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return rgb[y0 : y0 + s, x0 : x0 + s]


def iter_video_frames(path: str, size: int) -> Iterator[Tuple[int, Image.Image, float]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = _center_square_crop(rgb)
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
            yield idx, Image.fromarray(rgb), fps
            idx += 1
    finally:
        cap.release()


class VideoWriter:
    def __init__(self, path: str, size: int, fps: float):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._w = cv2.VideoWriter(path, fourcc, fps, (size, size))
        if not self._w.isOpened():
            raise RuntimeError(f"cannot open video writer: {path}")

    def write_pil(self, img: Image.Image) -> None:
        rgb = np.asarray(img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._w.write(bgr)

    def close(self) -> None:
        self._w.release()


@dataclass
class RunStats:
    timings: List[StepTiming] = field(default_factory=list)
    wall_start: float = 0.0
    wall_end: float = 0.0

    def report(self) -> str:
        if not self.timings:
            return "(no frames processed)"
        n = len(self.timings)

        def avg(attr):
            return sum(getattr(t, attr) for t in self.timings) / n

        avg_total = avg("total_ms")
        wall = (self.wall_end - self.wall_start) * 1000
        return (
            f"frames={n}  wall={wall:.0f}ms  fps_wall={n / (wall / 1000):.2f}\n"
            f"  per-frame avg ms:  total={avg_total:.1f}  "
            f"te={avg('te_ms'):.1f}  vae_enc={avg('vae_enc_ms'):.1f}  "
            f"denoise={avg('denoise_ms'):.1f}  vae_dec={avg('vae_dec_ms'):.1f}"
        )


def _log_frame(idx: int, t: StepTiming, prefix: str = "") -> None:
    print(
        f"{prefix}[{idx:04d}] total={t.total_ms:.0f}ms  "
        f"te={t.te_ms:.0f}  vae_enc={t.vae_enc_ms:.0f}  "
        f"denoise={t.denoise_ms:.0f}  vae_dec={t.vae_dec_ms:.0f}",
        flush=True,
    )


def _maybe_blend(
    out_pil: Image.Image, frame_pil: Image.Image, blender: Optional[OutputBlender],
) -> Image.Image:
    if blender is None or blender.alpha >= 1.0:
        return out_pil
    out_rgb = np.asarray(out_pil.convert("RGB"))
    in_rgb = np.asarray(frame_pil.convert("RGB"))
    blended = blender.apply(out_rgb, in_rgb)
    return Image.fromarray(blended)


def run_video(
    worker: EditWorker,
    in_path: str,
    out_path: str,
    size: int,
    log_every: int = 10,
    max_frames: Optional[int] = None,
    blender: Optional[OutputBlender] = None,
) -> RunStats:
    stats = RunStats()
    writer: Optional[VideoWriter] = None
    try:
        stats.wall_start = time.perf_counter()
        for idx, frame, fps in iter_video_frames(in_path, size):
            if max_frames is not None and idx >= max_frames:
                break
            out_img, t = worker.step(frame)
            out_img = _maybe_blend(out_img, frame, blender)
            if writer is None:
                writer = VideoWriter(out_path, size, fps)
            writer.write_pil(out_img)
            stats.timings.append(t)
            if log_every and idx % log_every == 0:
                _log_frame(idx, t)
        stats.wall_end = time.perf_counter()
    finally:
        if writer is not None:
            writer.close()
    return stats


def run_video_pipelined(
    worker: EditWorker,
    in_path: str,
    out_path: str,
    size: int,
    log_every: int = 10,
    max_frames: Optional[int] = None,
    blender: Optional[OutputBlender] = None,
) -> RunStats:
    """Pipelined runtime: TE/VAE_enc of frame n+1 overlaps with denoise/dec of frame n.

    Steady-state per-frame wall time ~= max(TE+VAE_enc, denoise+VAE_dec).
    """
    stats = RunStats()
    writer: Optional[VideoWriter] = None
    frame_iter = iter_video_frames(in_path, size)
    try:
        cur = next(frame_iter)
    except StopIteration:
        return stats
    prefetch = worker.prefetch(cur[1])  # bootstrap

    try:
        stats.wall_start = time.perf_counter()
        while True:
            cur_idx, cur_frame, cur_fps = cur
            if max_frames is not None and cur_idx >= max_frames:
                break
            try:
                nxt = next(frame_iter)
            except StopIteration:
                nxt = None

            # Kick prefetch for n+1 BEFORE running denoise n; the side
            # stream then runs concurrently with the default-stream denoise.
            next_prefetch = (
                worker.prefetch(nxt[1])
                if nxt is not None
                and (max_frames is None or nxt[0] < max_frames)
                else None
            )

            out_img, t = worker.step_with_prefetch(cur_frame, prefetch)
            out_img = _maybe_blend(out_img, cur_frame, blender)
            if writer is None:
                writer = VideoWriter(out_path, size, cur_fps)
            writer.write_pil(out_img)
            stats.timings.append(t)
            if log_every and cur_idx % log_every == 0:
                _log_frame(cur_idx, t, prefix="(P)")

            if nxt is None:
                break
            cur = nxt
            prefetch = next_prefetch
        stats.wall_end = time.perf_counter()
    finally:
        if writer is not None:
            writer.close()
    return stats
