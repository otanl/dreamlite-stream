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

from typing import List, Optional

import cv2
import numpy as np


def _blend_pair(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear blend at fractional time ``t`` in [0, 1] from a to b."""
    return cv2.addWeighted(a, 1.0 - t, b, t, 0)


def _flow_cheap(prev_bgr: np.ndarray, curr_bgr: np.ndarray,
                target_width: int = 256) -> np.ndarray:
    """Farneback optical flow downsampled to ~``target_width`` px wide
    for speed, then upsampled back to the original resolution with
    flow magnitudes scaled accordingly. Returns flow at original
    resolution with shape ``(H, W, 2)`` and dtype float32.

    At target_width=256 on a 1024-wide frame this is ~5 ms per call;
    full-resolution Farneback would be ~25 ms. The accuracy loss is
    not visible at the alpha-blend strengths we use (0.3–0.6).
    """
    H, W = prev_bgr.shape[:2]
    scale = target_width / W if W > target_width else 1.0
    if scale < 1.0:
        new_w = int(W * scale)
        new_h = int(H * scale)
        prev_small = cv2.resize(prev_bgr, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_bgr, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)
    else:
        prev_small, curr_small = prev_bgr, curr_bgr
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
    flow_small = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0,
    )
    if scale < 1.0:
        flow = cv2.resize(flow_small, (W, H),
                          interpolation=cv2.INTER_LINEAR) / scale
    else:
        flow = flow_small
    return flow.astype(np.float32)


def _warp_with_flow(img_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp ``img_bgr`` so pixel at (x, y) is taken from
    ``img_bgr[y - flow_y, x - flow_x]`` (i.e. ``img_bgr`` is the
    predecessor and ``flow`` is the forward flow from predecessor to
    successor). Uses cv2.remap with BORDER_REPLICATE so disocclusion
    edges stretch rather than show black."""
    H, W = flow.shape[:2]
    xs, ys = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )
    return cv2.remap(
        img_bgr,
        xs - flow[..., 0],
        ys - flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ---------------------------------------------------------------------
# RIFE-based interpolation (Tier 1-A v1)
# ---------------------------------------------------------------------
# Wraps Practical-RIFE (https://github.com/hzwer/Practical-RIFE, MIT)
# as a drop-in replacement for the linear ``expand_linear``.
#
# Caller must (a) clone Practical-RIFE somewhere accessible and (b) have
# a downloaded RIFE_HDv3 checkpoint (or compatible). The model object is
# cached at module level after first load.

_rife_model = None  # module-level cache


def _load_rife(model_path: str, repo_path: Optional[str] = None,
               device: str = "cuda"):
    """Lazy-load a Practical-RIFE model.

    Args:
        model_path: filesystem path to either the ``.pkl`` checkpoint or
            its containing directory (Practical-RIFE's load_model API
            takes the directory).
        repo_path: directory containing a Practical-RIFE checkout. Added
            to sys.path so ``from model.RIFE_HDv3 import Model`` works.
            Pass None if Practical-RIFE is already importable.
        device: target CUDA device.

    Returns:
        Loaded model instance, cached at module level.
    """
    global _rife_model
    if _rife_model is not None:
        return _rife_model

    import sys
    from pathlib import Path

    if repo_path:
        sys.path.insert(0, str(Path(repo_path).resolve()))

    try:
        from model.RIFE_HDv3 import Model  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "Could not import Practical-RIFE. Pass --rife_path PATH "
            "where PATH is a clone of "
            "https://github.com/hzwer/Practical-RIFE (or install it). "
            f"Underlying error: {e}"
        ) from e

    import torch  # noqa: F401  (imported so the model finds CUDA)

    model = Model()
    # load_model expects a DIRECTORY containing train_log/RIFE_HDv3.pkl
    # by default; accept either form from the user.
    mp = Path(model_path)
    if mp.is_file():
        # User pointed at the .pkl directly; load_model wants the parent.
        model.load_model(str(mp.parent), -1)
    else:
        model.load_model(str(mp), -1)
    model.eval()
    if hasattr(model, "device"):
        try:
            model.device()
        except Exception:
            pass

    _rife_model = model
    return _rife_model


def _bgr_to_rife_tensor(bgr: np.ndarray, device: str = "cuda"):
    """HxWx3 uint8 BGR → (1, 3, H, W) float32 in [0, 1] on device.
    Pads H and W to the next multiple of 32 (RIFE requires it)."""
    import torch
    import torch.nn.functional as F

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    h, w = t.shape[-2:]
    ph = (32 - h % 32) % 32
    pw = (32 - w % 32) % 32
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="replicate")
    return t.to(device), (h, w)


def _rife_tensor_to_bgr(t, orig_hw) -> np.ndarray:
    """(1, 3, Hp, Wp) float32 → HxWx3 uint8 BGR, cropping the padding."""
    h, w = orig_hw
    rgb = (
        t[:, :, :h, :w].clamp(0, 1).mul_(255.0).byte().cpu().numpy()[0]
        .transpose(1, 2, 0)
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def expand_rife(bgr_frames: List[np.ndarray], factor: int, model,
                device: str = "cuda") -> List[np.ndarray]:
    """RIFE-based interpolation. Supports ``factor`` in {2, 4} via
    recursive halving; other factors fall back to a single RIFE
    midpoint per pair (i.e. effectively factor=2 regardless).

    Quality is noticeably better than ``expand_linear`` on fast
    non-rigid motion; cost is ~5–15 ms per pair on a 3090 Ti at 512².
    """
    if factor < 2 or len(bgr_frames) < 2:
        return list(bgr_frames)

    import torch

    out: List[np.ndarray] = [bgr_frames[0]]
    with torch.no_grad():
        for i in range(len(bgr_frames) - 1):
            a, b = bgr_frames[i], bgr_frames[i + 1]
            ta, hwa = _bgr_to_rife_tensor(a, device)
            tb, _ = _bgr_to_rife_tensor(b, device)
            mid_t = model.inference(ta, tb, 1.0)
            mid = _rife_tensor_to_bgr(mid_t, hwa)

            if factor == 4:
                tm, _ = _bgr_to_rife_tensor(mid, device)
                left_t = model.inference(ta, tm, 1.0)
                right_t = model.inference(tm, tb, 1.0)
                out.append(_rife_tensor_to_bgr(left_t, hwa))
                out.append(mid)
                out.append(_rife_tensor_to_bgr(right_t, hwa))
            else:
                # factor=2 or unsupported → single midpoint.
                out.append(mid)
            out.append(b)
    return out


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


def temporal_blend(
    bgr_frames: List[np.ndarray],
    prev_anchor: Optional[np.ndarray],
    alpha: float,
    use_flow_warp: bool = False,
) -> List[np.ndarray]:
    """Sequential temporal alpha blend: each output frame is mixed with
    its (already-blended) predecessor by ``alpha``.

    ``frame[i] = (1 - alpha) * out[i] + alpha * ref``

    where ``ref`` is either:
      - the previous blended frame as-is (``use_flow_warp=False``,
        v0): simple, fast, but motion-blur ghosts on fast content;
      - the previous blended frame warped via Farneback flow toward
        ``out[i]`` (``use_flow_warp=True``, v1): removes most motion
        ghosting at the cost of ~5 ms per pair on CPU and occasional
        warp errors at disocclusion edges.

    ``prev_anchor`` seeds the predecessor for the first frame; pass the
    last blended frame from the previous batch to keep continuity
    across batch boundaries.

    ``alpha`` is clipped to [0, 0.95]; 0 returns input unchanged.
    """
    if alpha <= 0.0 or not bgr_frames:
        return list(bgr_frames)
    alpha = float(min(max(alpha, 0.0), 0.95))
    out: List[np.ndarray] = []
    prev = prev_anchor if prev_anchor is not None else bgr_frames[0]
    for f in bgr_frames:
        if use_flow_warp and prev is not f:
            flow = _flow_cheap(prev, f)
            ref = _warp_with_flow(prev, flow)
        else:
            ref = prev
        b = cv2.addWeighted(f, 1.0 - alpha, ref, alpha, 0)
        out.append(b)
        prev = b
    return out
