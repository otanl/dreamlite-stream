"""Frame interpolation for the live demo.

The pipeline emits one stylized frame per source frame at the pipeline
fps (~25 fps end-to-end on a 3090 Ti at B=8). Linear blending between
consecutive output frames inserts cheap intermediates so the displayed
fps doubles or quadruples without invoking the model again — useful
purely for demo smoothness.

V0 = linear blend, no external dependency, ghosts mildly on fast
motion but is acceptable for the 30 fps source rate the demo runs at.

For higher motion fidelity, see ``notes/roadmap.md`` (Tier 1 A): drop
in a RIFE call inside ``_blend_pair`` and the rest of the pipeline
stays unchanged.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def _blend_pair(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear blend at fractional time ``t`` in [0, 1] from a to b."""
    return cv2.addWeighted(a, 1.0 - t, b, t, 0)


def expand_linear(bgr_frames: List[np.ndarray], factor: int) -> List[np.ndarray]:
    """Insert ``factor-1`` linearly-blended intermediates between each
    consecutive pair in ``bgr_frames``.

    Args:
        bgr_frames: list of HxWx3 uint8 BGR images.
        factor: 1 returns the input unchanged; 2 doubles frame count
            (one midpoint per pair); 4 inserts three intermediates per
            pair.

    Returns:
        Expanded list of length ``factor * N - (factor - 1)`` where
        ``N == len(bgr_frames)``.
    """
    if factor < 2 or len(bgr_frames) < 2:
        return list(bgr_frames)
    out: List[np.ndarray] = [bgr_frames[0]]
    for i in range(len(bgr_frames) - 1):
        a = bgr_frames[i]
        b = bgr_frames[i + 1]
        for k in range(1, factor):
            t = k / factor
            out.append(_blend_pair(a, b, t))
        out.append(b)
    return out
