"""Live webcam stylization demo.

Captures from a webcam (or an NDI source — e.g. TouchDesigner via
``--ndi_source``), runs each B-frame batch through the DreamLite-mobile +
Qwen3-VL + Temporal LLLite v3 pipeline, and shows input | output
side-by-side in an OpenCV window.

Architecture (all single-process, three threads):

  - Camera/NDI thread (``camera_loop`` or ``ndi_camera_loop``):
    Continuously pushes (PIL, BGR) frames into a small bounded
    ``cam_q`` (maxsize = B+1). Drop-oldest on overflow so the queue
    always reflects the most recent ~1 batch of source frames.

  - Pipeline thread (``pipeline_loop``):
    Drains ``cam_q`` (keeping only the freshest B frames each iter),
    submits the NEXT-NEXT batch's CPU prep (and, on a TE refresh
    boundary, the full Q3-VL TE forward on a dedicated te_side_stream)
    to ``cpu_prep_pool``. Then waits on the previous iter's cpu_prep,
    kicks VAE-encode for the next batch on the main side_stream, and
    runs ``step_batch_with_prefetch`` for the current batch on the main
    CUDA stream. Compiled UNet replays under ``mode="reduce-overhead"``;
    cuBLAS/cuDNN are pre-warmed on this thread to avoid lazy-init mid
    graph capture.

  - Prompt-input thread (``prompt_reader_loop``):
    Background stdin reader. Whenever the user types a new prompt and
    presses Enter, ``state.prompt`` is updated and ``worker._force_te_-
    refresh`` is set so the next bg cpu_prep recomputes the TE cache.
    The pipeline is never paused for input.

  - Display loop (main thread):
    Pops finished batches from a maxsize-1 ``display_q`` (drop-oldest
    on the pipeline's ``put`` so display lag never feeds back into
    pipeline iter_dur), draws an fps overlay, and paces frame-by-frame
    via ``cv2.waitKey``.

Performance optimisations beyond the paper's batched eval:

  - ``--te_refresh_every N``: cache Q3-VL prompt_embeds across N
    batches; only refresh image-aware TE periodically. Eliminates the
    ~250 ms TE-forward cost on the cached majority of iterations.
  - ``--te_batch_one``: when refreshing, run Q3-VL on a single
    representative frame and broadcast the result to the full batch.
    Shrinks the multimodal sequence ~B× and the refresh GPU wall
    correspondingly.
  - TE forward lives on its own ``_te_side_stream`` driven from the bg
    cpu_prep pool, so the ``.item()`` sync inside Q3-VL's rot_pos_emb
    blocks the bg thread (not the main pipeline thread) and runs
    concurrently with the previous batch's main-stream step.
  - ``padding="max_length", max_length=256`` for the processor: keeps
    encoder_hidden_states' sequence length constant across prompts so
    a live prompt change doesn't force torch.compile to re-record the
    captured CUDA graph mid-flight.

Latency: at source 30 fps with B=8 there is a one-batch-deep pipeline
(~270 ms) plus the cam_q/display_q buffer (~1 batch combined). Live
camera→display latency is around ~700–900 ms. This is the streaming-
throughput operating point reported in the paper, not zero-latency
interactive.

Controls (focus the demo window):
  q        quit
  space    pause / resume (both pipeline and display)
  s        save current output frame to <save_dir>/demo_capture_*.png
  p        print a hint that prompts are typed in the launching terminal

Prompt changes: type into the launching terminal (look for ``prompt>``)
and press Enter — the pipeline keeps running and the new style appears
~1–2 iters later.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
from pathlib import Path

import cv2
import numpy as np
import torch
import torch._dynamo
from PIL import Image

torch._dynamo.config.cache_size_limit = 256

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
_LLLITE = _ROOT.parent / "dreamlite-lllite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))
sys.path.insert(0, str(_LLLITE / "src"))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402
from dreamlite_lllite import apply_lllite  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from dreamlite_stream import BatchedEditWorker, SharedState  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--lllite_weights",
                   default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cond_refresh_every", type=int, default=8)
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true", default=True,
                   help="enable torch.compile (requires triton; eager fallback on Windows)")
    p.add_argument("--no_compile", dest="compile", action="store_false")
    p.add_argument("--compile_mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help='torch.compile mode. "reduce-overhead" uses CUDA Graphs '
                        '(faster single-threaded) but breaks with the demo\'s '
                        'concurrent prefetch+step pattern; use "default" here.')
    p.add_argument("--quant_te", default="none",
                   choices=["none", "nf4", "fp4", "int8"],
                   help='Quantize the Q3-VL TE on load to reduce prefetch wall '
                        '(deployment path from paper §sec:deploy). Needs '
                        'bitsandbytes installed. "nf4" is the recommended choice.')
    p.add_argument("--ndi_source", default=None,
                   help='NDI source name pattern (substring match, e.g. "td" '
                        'to match "DESKTOP (td)"). When set, the demo receives '
                        'frames over NDI instead of opening a webcam.')
    p.add_argument("--ndi_extra_ips", default=None,
                   help='[deprecated] Use ~/.ndi/ndi-config.v1.json for NDI '
                        'discovery server config on Linux. Kept for back-compat; '
                        'no-op in current implementation.')
    p.add_argument("--te_refresh_every", type=int, default=1,
                   help='Recompute Q3-VL prompt_embeds only every N batches; '
                        'reuse cached embeds in between. At B=8 the TE forward '
                        'is ~250ms per batch on a 3090Ti — caching amortises '
                        'that, similar in spirit to --cond_refresh_every. '
                        '1 = no caching, 8 = refresh once per second at 30fps.')
    p.add_argument("--te_batch_one", action="store_true", default=False,
                   help='On TE refresh, run Q3-VL on a single representative '
                        'frame (frames[0]) and broadcast embeds to the full '
                        'batch. Shrinks the refresh GPU wall ~B× (multimodal '
                        'sequence shortens). Safe with --te_refresh_every>1 '
                        "since per-frame image-aware conditioning is already "
                        'being amortised by the cache.')
    p.add_argument("--fixed_noise", action="store_true", default=False,
                   help='Use the SAME init noise pattern for every frame '
                        '(seed_offset stays at 0 instead of advancing with '
                        's.frame_idx). Removes per-frame stochastic variation '
                        "in the denoise initial latents, so per-frame output "
                        'drift comes purely from input changes — useful for '
                        'isolating whether observed flicker is from input '
                        'sensor noise or from the diffusion noise pattern.')
    p.add_argument("--interp_factor", type=int, default=1,
                   help='Display-side frame interpolation factor. 1 = off; '
                        '2 inserts one intermediate frame between each pair '
                        'of pipeline outputs (doubles displayed fps); 4 '
                        'inserts three (quadruples displayed fps). The '
                        'method (linear or RIFE) is chosen by '
                        '--interp_method. Pipeline throughput is unchanged.')
    p.add_argument("--interp_method", default="linear",
                   choices=["linear", "rife"],
                   help='Interpolation method used when --interp_factor > 1. '
                        '"linear" (default) blends pixels; cheap, mild '
                        'ghosting on fast motion. "rife" calls Practical-'
                        'RIFE on each pair (better motion quality, ~5-15 ms '
                        'per pair on a 3090 Ti). For "rife" you must also '
                        'pass --rife_path (and optionally --rife_model).')
    p.add_argument("--rife_path", default=None,
                   help='Directory of a Practical-RIFE checkout '
                        '(https://github.com/hzwer/Practical-RIFE), added to '
                        'sys.path so the model module is importable. '
                        'Only used when --interp_method=rife.')
    p.add_argument("--rife_model", default=None,
                   help='Path to a Practical-RIFE checkpoint directory or '
                        '.pkl file (e.g. train_log/RIFE_HDv3.pkl from the '
                        'Practical-RIFE Google-Drive download). If omitted, '
                        'defaults to <rife_path>/train_log.')
    p.add_argument("--temporal_blend_alpha", type=float, default=0.0,
                   help='Display-side temporal alpha blend: smooths each '
                        'output toward its predecessor by alpha in [0, 0.95]. '
                        '0 = off (default). Reduces flicker on static '
                        'regions at the cost of motion-blur ghosting on '
                        'fast content. Reasonable values: 0.3-0.6. Applied '
                        'before --interp_factor so interpolation runs on '
                        'the smoothed sequence.')
    p.add_argument("--temporal_blend_warp", action="store_true", default=False,
                   help='Use Farneback flow to warp the predecessor frame '
                        'toward the current one before blending. Removes '
                        'most motion-blur ghosting at the cost of ~5 ms per '
                        'frame on CPU and occasional warp errors at '
                        'disocclusion edges. Only meaningful when '
                        '--temporal_blend_alpha > 0.')
    p.add_argument("--save_dir", default=str(_ROOT / "out" / "demo"))
    p.add_argument("--verbose_timing", action="store_true",
                   help="Print per-batch collect/prefetch/step timings to console")
    return p.parse_args()


def load_pipeline_with_quant_te(model_path: str, quant: str, device: str):
    """Load DreamLiteMobilePipeline, optionally swapping the Q3-VL TE for a
    bitsandbytes-quantized version (NF4 / FP4 / int8). Mirrors the recipe in
    scripts/test_te_4bit.py used for the paper's §sec:deploy measurement.

    Returns (pipeline, runtime_dtype). When quantization is on we use float16
    throughout (matches the bnb compute_dtype convention).
    """
    if quant == "none":
        pipe = DreamLiteMobilePipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
        ).to(device)
        return pipe, torch.bfloat16

    try:
        from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    except ImportError as e:
        raise RuntimeError(
            f"--quant_te {quant} requires bitsandbytes + a Qwen3-VL "
            f"transformers class. Install: `pip install bitsandbytes`. "
            f"Import error: {e}"
        )

    if quant == "int8":
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
    else:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=quant,
            bnb_4bit_use_double_quant=True,
        )

    print(f"[quant] pipeline (TE quant={quant}); other components in fp16")
    pipe = DreamLiteMobilePipeline.from_pretrained(
        model_path, torch_dtype=torch.float16,
    )
    te_path = Path(model_path) / "text_encoder"
    print(f"[quant] re-loading TE from {te_path} with bnb {quant}...")
    te_quant = Qwen3VLForConditionalGeneration.from_pretrained(
        te_path,
        quantization_config=qconfig,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    del pipe.text_encoder
    pipe.text_encoder = te_quant
    pipe.vae = pipe.vae.to(device)
    pipe.unet = pipe.unet.to(device)
    return pipe, torch.float16


def crop_resize_to_square(bgr: np.ndarray, size: int) -> np.ndarray:
    h, w, _ = bgr.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    bgr = bgr[y0 : y0 + s, x0 : x0 + s]
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)


def overlay_text(img_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img_bgr.copy()
    pad = 8
    line_h = 22
    box_h = pad + line_h * len(lines) + pad // 2
    box_w = max(180, max(len(l) for l in lines) * 9 + 2 * pad)
    cv2.rectangle(out, (0, 0), (box_w, box_h), (0, 0, 0), thickness=-1)
    for i, line in enumerate(lines):
        cv2.putText(
            out, line, (pad, pad + line_h * (i + 1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA,
        )
    return out


def collect_batch(cap, B: int, size: int):
    """Read B frames from `cap`. Returns (pil_frames, bgr_squares)."""
    pils, bgrs = [], []
    while len(pils) < B:
        ok, bgr = cap.read()
        if not ok:
            time.sleep(0.005)
            continue
        bgr_sq = crop_resize_to_square(bgr, size)
        rgb_sq = cv2.cvtColor(bgr_sq, cv2.COLOR_BGR2RGB)
        pils.append(Image.fromarray(rgb_sq))
        bgrs.append(bgr_sq)
    return pils, bgrs


def ndi_camera_loop(
    ndi_source_pattern: str, _unused_extra_ips: str,
    cam_q: Queue, args,
    stop_event: threading.Event,
    pause_event: threading.Event,
):
    """NDI producer thread (uses cyndilib, NDI SDK 6 compatible).

    Discovery on Linux: the NDI library reads ``~/.ndi/ndi-config.v1.json``
    (env vars like NDI_RECV_DISCOVERY_SERVERS are a Windows-side path; on
    Linux you configure via JSON). For WSL2 mirrored mode the config to
    use a discovery server running on the same machine is::

        ~/.ndi/ndi-config.v1.json:
        {"ndi": {"networks": {"discovery": "127.0.0.1"}}}

    Run ``ndi-discovery-server`` (from the NDI SDK Linux ``bin/``) in
    parallel; the sender (TouchDesigner / OBS DistroAV) must also be
    configured (NDI Access Manager) to use the same discovery server."""
    try:
        from cyndilib.finder import Finder
        from cyndilib.receiver import Receiver
        from cyndilib.video_frame import VideoFrameSync
        from cyndilib.wrapper import RecvBandwidth, RecvColorFormat
    except ImportError as e:
        print(f"[ndi] cyndilib not installed: {e}\n      pip install cyndilib")
        stop_event.set()
        return

    finder = Finder()
    finder.open()
    target = None
    for _ in range(60):  # up to ~6s
        if stop_event.is_set():
            finder.close()
            return
        time.sleep(0.1)
        names = finder.get_source_names()
        for n in names:
            if ndi_source_pattern.lower() in n.lower():
                try:
                    target = finder.get_source(n)
                except Exception:
                    continue
                break
        if target is not None:
            break

    if target is None:
        print(f"[ndi] no source matching {ndi_source_pattern!r}. "
              f"visible: {finder.get_source_names()}\n"
              f"      (hint: ensure ~/.ndi/ndi-config.v1.json points at "
              f"a discovery server that the sender also uses)")
        finder.close()
        stop_event.set()
        return

    print(f"[ndi] connected: {target.name}")

    recv = Receiver(
        color_format=RecvColorFormat.BGRX_BGRA,
        bandwidth=RecvBandwidth.highest,
        recv_name="dreamlite-stream-demo",
    )
    recv.set_source(target)

    video_frame = VideoFrameSync()
    recv.frame_sync.set_video_frame(video_frame)
    _first_frame_logged = False

    try:
        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.05)
                continue

            recv.frame_sync.capture_video()
            arr = video_frame.get_array()
            if arr is None or arr.size == 0:
                time.sleep(0.005)
                continue

            # cyndilib get_array() returns a flat 1D uint8 buffer; reshape
            # using the frame's xres/yres. For BGRX_BGRA color format there
            # are 4 channels per pixel.
            xres = video_frame.xres
            yres = video_frame.yres
            if xres <= 0 or yres <= 0:
                time.sleep(0.005)
                continue
            expected = yres * xres * 4
            if arr.size != expected:
                if not _first_frame_logged:
                    print(f"[ndi] unexpected frame size: got {arr.size}, "
                          f"expected {expected} for {xres}x{yres}x4")
                    _first_frame_logged = True
                time.sleep(0.005)
                continue
            bgra = arr.reshape(yres, xres, 4)
            bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            if not _first_frame_logged:
                print(f"[ndi] first frame: {xres}x{yres}")
                _first_frame_logged = True

            bgr_sq = crop_resize_to_square(bgr, args.size)
            rgb_sq = cv2.cvtColor(bgr_sq, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb_sq)
            try:
                cam_q.put((pil, bgr_sq), timeout=0.05)
            except Full:
                try:
                    cam_q.get_nowait()
                    cam_q.put_nowait((pil, bgr_sq))
                except (Empty, Full):
                    pass
            # Release GIL aggressively so the pipeline thread has CPU.
            # Source is 30fps so this loop only needs to run ~30 times/sec.
            time.sleep(0.020)
    finally:
        try:
            recv.disconnect()
        except Exception:
            pass
        finder.close()


def camera_loop(
    cap, cam_q: Queue, args,
    stop_event: threading.Event,
    pause_event: threading.Event,
):
    """Producer thread: always reads camera, pushes latest PIL+BGR frames
    onto a bounded queue. If the consumer (pipeline) falls behind, the
    oldest queued frame is dropped so we never accumulate stale motion."""
    while not stop_event.is_set():
        if pause_event.is_set():
            cap.grab()  # discard one frame
            time.sleep(0.03)
            continue
        ok, bgr = cap.read()
        if not ok:
            time.sleep(0.005)
            continue
        bgr_sq = crop_resize_to_square(bgr, args.size)
        rgb_sq = cv2.cvtColor(bgr_sq, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb_sq)
        try:
            cam_q.put((pil, bgr_sq), timeout=0.05)
        except Full:
            try:
                cam_q.get_nowait()  # drop oldest
                cam_q.put_nowait((pil, bgr_sq))
            except (Empty, Full):
                pass


def prompt_reader_loop(args, state, worker, stop_event: threading.Event):
    """Background stdin reader: lets the user type a new prompt and press
    Enter at any time without pausing the demo. The next bg cpu_prep
    call picks it up via ``worker._force_te_refresh`` and the new style
    appears 1-2 iters later (~600 ms). Empty lines are ignored. EOF
    (e.g. piped input ending) just exits the reader, not the demo."""
    bar = "=" * 64
    print(
        f"\n{bar}\n"
        f"[prompt input] Click THIS TERMINAL (not the demo window), then\n"
        f"[prompt input] type a new prompt and press Enter — pipeline\n"
        f"[prompt input] keeps running. Empty line = ignored.\n"
        f"{bar}\n",
        flush=True,
    )
    while not stop_event.is_set():
        try:
            line = input("prompt> ").strip()
        except EOFError:
            print("[prompt input] EOF on stdin; reader exiting", flush=True)
            return
        except Exception as e:
            print(f"[prompt input] reader error: {e!r}", flush=True)
            return
        if not line:
            continue
        args.prompt = line
        state.prompt = line
        worker._force_te_refresh = True
        worker._cond_call_idx = 0
        print(
            f"[prompt] applied; new style visible after ~1-2 iters: {line!r}",
            flush=True,
        )


def collect_from_cam_q(cam_q: Queue, B: int, stop_event: threading.Event):
    """Pull at least B frames, then drain any extras non-blockingly and
    keep only the most recent B. Reduces camera→display latency by
    skipping ahead when the pipeline can't keep up with the source rate
    — older buffered frames are dropped rather than processed. Acceptable
    here because the demo prioritises responsiveness over preserving
    every camera frame.

    Prints a warning if no frame arrives for >3 s — usually means the
    NDI source dropped (TouchDesigner lost focus, network blip, etc.).
    The pipeline keeps trying so the demo recovers when frames resume,
    but the warning surfaces the issue to the user.
    """
    pils, bgrs = [], []
    t0 = time.perf_counter()
    last_warn = t0
    while len(pils) < B and not stop_event.is_set():
        try:
            pil, bgr = cam_q.get(timeout=0.5)
            pils.append(pil)
            bgrs.append(bgr)
        except Empty:
            now = time.perf_counter()
            if now - last_warn > 3.0:
                print(
                    f"[cam_q] no frames for {now - t0:.1f}s — NDI source "
                    "may have dropped. Pipeline is stuck on collect.",
                    flush=True,
                )
                last_warn = now
            continue
    # Drain anything else already buffered so we end up with the freshest
    # B frames; older ones are discarded.
    try:
        while True:
            pil, bgr = cam_q.get_nowait()
            pils.append(pil)
            bgrs.append(bgr)
    except Empty:
        pass
    if len(pils) > B:
        pils = pils[-B:]
        bgrs = bgrs[-B:]
    return pils, bgrs


def pipeline_loop(
    worker, args,
    primed_state,
    cam_q: Queue,
    display_q: Queue,
    stop_event: threading.Event,
    pause_event: threading.Event,
):
    """Background thread that keeps producing batches into display_q.

    Two-stage pipeline:

      1. A CPU-prep pool thread runs `worker.prefetch_cpu_only` for the
         NEXT-NEXT batch — pure Python/HF preprocessing, no CUDA work.
         Touches no CUDA so it cannot invalidate the main thread's
         CUDA-graph capture/replay.

      2. The main pipeline thread (this function) does on every iter:
           a) take the already-prepared cpu_prep from a queue,
           b) call `worker.prefetch_gpu_kick(cpu_prep, frames)` to queue
              TE + VAE-encode on the side CUDA stream (returns instantly
              after the kernel launches),
           c) call `worker.step_batch_with_prefetch` for the CURRENT batch
              on the main stream (~270 ms blocking on the GPU).

      The CPU prep thread is always ~1-2 batches ahead, so step (a) is
      instant in steady state. Per-iter wall on the main thread is then
      dominated by step alone (~270 ms ⇒ ~30 fps at B=8), matching the
      paper's per-batch wall.
    """
    # Warm up cuBLAS handle + workspace + cuDNN on this thread before any
    # compiled CUDA-graph capture lands here. Background:
    # torch.compile mode=reduce-overhead defers cudagraphify to the first
    # call after compile. On a steady-state cache-hit iter, the prior
    # gpu_kick only runs VAE-encode (cuDNN-only) on this thread, so
    # cuBLAS hasn't been initialized yet. The first addmm inside the
    # captured UNet then lazy-inits cuBLAS mid-capture, which CUDA refuses
    # (cublasCreate / workspace alloc are not capture-safe) and raises
    # CUBLAS_STATUS_NOT_INITIALIZED → cudaErrorStreamCaptureInvalidated.
    # Touching the relevant backends up front with representatively-sized
    # ops forces handle creation + workspace allocation now.
    with torch.no_grad():
        # cuBLAS: matmul at a size big enough to provoke any GEMM
        # workspace allocation cuBLAS may need lazily.
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        _ = a @ a
        # cuDNN: a representative bf16 conv (UNet uses these heavily).
        x = torch.randn(args.batch_size, 64, 64, 64, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 64, 3, 3, device="cuda", dtype=torch.bfloat16)
        _ = torch.nn.functional.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        del a, _, x, w

    # Push the already-computed first batch outputs
    try:
        display_q.put((
            primed_state["bgrs_cur"],
            primed_state["first_outputs"],
            primed_state["first_batch_dur"],
        ), timeout=2.0)
    except Full:
        return

    buf_cur = primed_state["buf_next"]
    bgrs_cur = primed_state["bgrs_next"]
    pf_cur = primed_state["pf_next"]
    batch_idx = 1

    # Background CPU-prep thread. Runs `worker.prefetch_cpu_only`
    # (pure CPU, no CUDA) concurrently with main-thread GPU work.
    cpu_prep_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpuprep")

    # Bootstrap state for the two-stage pipeline. We received from main():
    #   buf_cur, pf_cur  — batch about to be step'd (pf already kicked)
    # We need to set up the "next" slot so iter 0 has it ready.
    last_iter_end = time.perf_counter()
    try:
        buf_next, bgrs_next = collect_from_cam_q(cam_q, args.batch_size, stop_event)
    except Exception:
        cpu_prep_pool.shutdown(wait=False)
        return
    if stop_event.is_set():
        cpu_prep_pool.shutdown(wait=False)
        return
    # Submit cpu_prep for `buf_next`. By the time iter 0 runs (after
    # collecting buf_after_next from the camera, which itself blocks
    # ~270ms at source rate), this CPU prep will be done.
    cpu_prep_future = cpu_prep_pool.submit(worker.prefetch_cpu_only, buf_next)

    try:
        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.05)
                last_iter_end = time.perf_counter()
                continue

            iter_t0 = time.perf_counter()

            # 1) Collect frames for the batch we'll step TWO iters from now.
            #    We submit its CPU prep so it's ready for the NEXT iter's
            #    GPU kick. This collect is camera-rate-bound (~270ms at
            #    30 fps for B=8), which is our natural inter-batch tick.
            tc0 = time.perf_counter()
            buf_after_next, bgrs_after_next = collect_from_cam_q(
                cam_q, args.batch_size, stop_event,
            )
            tc1 = time.perf_counter()
            if stop_event.is_set():
                break
            next_cpu_prep_future = cpu_prep_pool.submit(
                worker.prefetch_cpu_only, buf_after_next,
            )

            # 2) Wait for `buf_next`'s CPU prep (submitted in the previous
            #    iter or bootstrap). Should be done already in steady state.
            twcpu0 = time.perf_counter()
            try:
                # Generous timeout: a TE refresh under contention can take
                # ~500 ms, prompt-change can stack two refreshes, so 10 s
                # is firmly past "everything's fine" but well before "this
                # is hung forever". Hitting it means the bg thread is
                # stuck on a CUDA call.
                cpu_prep_next = cpu_prep_future.result(timeout=10.0)
            except concurrent.futures.TimeoutError:
                print(
                    f"[cpu_prep timeout] bg thread stuck >10s at iter "
                    f"{batch_idx}. Likely CUDA hang triggered by the most "
                    "recent prompt change (possible reasons: new prompt "
                    "tokenises to a different length than priming did, "
                    "forcing torch.compile to re-record the captured graph "
                    "on this thread). Stopping pipeline.",
                    flush=True,
                )
                stop_event.set()
                break
            except Exception as e:
                import traceback
                print(f"[cpu_prep error] {type(e).__name__}: {e!r}", flush=True)
                traceback.print_exc()
                stop_event.set()
                break
            twcpu1 = time.perf_counter()

            # 3) GPU-kick `buf_next` on the side stream (Python wall ~ms).
            tk0 = time.perf_counter()
            try:
                pf_next = worker.prefetch_gpu_kick(cpu_prep_next, frames=buf_next)
            except Exception as e:
                import traceback
                print(f"[kick error] {type(e).__name__}: {e!r}", flush=True)
                traceback.print_exc()
                stop_event.set()
                break
            tk1 = time.perf_counter()

            # 4) Step the CURRENT batch on the main stream. Main waits on
            #    pf_cur's side-stream done event (already complete in
            #    steady state) then runs UNet + VAE-decode (~270ms B=8).
            t0 = time.perf_counter()
            try:
                outputs, _ = worker.step_batch_with_prefetch(buf_cur, pf_cur)
            except Exception as e:
                import traceback
                print(f"[pipeline error] {type(e).__name__}: {e!r}", flush=True)
                traceback.print_exc()
                stop_event.set()
                break
            t1 = time.perf_counter()
            step_dur = t1 - t0

            iter_t1 = time.perf_counter()
            iter_dur = iter_t1 - last_iter_end
            last_iter_end = iter_t1

            # Non-blocking put with drop-oldest: if the display is behind,
            # we discard the OLDEST queued output batch rather than block
            # the pipeline. Blocking on put inflates iter_dur (because
            # last_iter_end is set before this call), which feeds back into
            # fps_smoothed → per_frame_ms grows → display gets even slower
            # → runaway. Dropping keeps the pipeline at steady state.
            try:
                display_q.put_nowait((bgrs_cur, outputs, iter_dur))
            except Full:
                try:
                    display_q.get_nowait()
                    display_q.put_nowait((bgrs_cur, outputs, iter_dur))
                except (Empty, Full):
                    pass

            if getattr(args, "verbose_timing", False):
                print(f"[t] batch {batch_idx:4d}  collect={1000*(tc1-tc0):5.0f}ms  "
                      f"wait_cpu_prep={1000*(twcpu1-twcpu0):5.0f}ms  "
                      f"gpu_kick={1000*(tk1-tk0):5.0f}ms  "
                      f"step={1000*step_dur:5.0f}ms  "
                      f"iter_total={1000*iter_dur:5.0f}ms  "
                      f"cam_q={cam_q.qsize()}  display_q={display_q.qsize()}",
                      flush=True)
            batch_idx += 1

            # Advance the slots.
            buf_cur, bgrs_cur, pf_cur = buf_next, bgrs_next, pf_next
            buf_next, bgrs_next = buf_after_next, bgrs_after_next
            cpu_prep_future = next_cpu_prep_future
    finally:
        cpu_prep_pool.shutdown(wait=False)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    pipeline, runtime_dtype = load_pipeline_with_quant_te(
        args.model, args.quant_te, args.device,
    )

    vae_downsample = 2 ** (len(pipeline.vae.config.encoder_block_out_channels) - 1)
    latent_hw = args.size // vae_downsample
    block_filter = [s.strip() for s in args.lllite_blocks.split(",")] if args.lllite_blocks else None
    controller = apply_lllite(
        pipeline.unet, cond_emb_dim=32, mlp_dim=64,
        cond_image_size=args.size, sample_size=latent_hw,
        inference_mode=True, max_batch_size=args.batch_size,
        block_filter=block_filter,
    )
    sd = load_file(args.lllite_weights)
    controller.load_state_dict(sd, strict=False)
    controller.to(device=args.device, dtype=runtime_dtype)
    controller.eval()
    controller.set_multiplier(1.0)
    print(f"[lllite] {len(controller.modules_dict)} hooks ({args.lllite_blocks}), "
          f"dtype={runtime_dtype}")

    state = SharedState(
        height=args.size, width=args.size,
        num_inference_steps=args.steps, prompt=args.prompt,
    )
    worker = BatchedEditWorker(
        pipeline=pipeline, state=state, batch_size=args.batch_size,
        device=args.device, dtype=runtime_dtype, seed=args.seed,
        compile=args.compile, compile_mode=args.compile_mode,
        lllite_controller=controller,
        cond_refresh_every=args.cond_refresh_every,
        te_refresh_every=args.te_refresh_every,
        te_batch_one=args.te_batch_one,
        fixed_noise=args.fixed_noise,
        cond_flow_workers=8,
    )

    use_ndi = args.ndi_source is not None
    if use_ndi:
        cap = None  # NDI receiver replaces VideoCapture
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera index {args.camera}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win_name = "dreamlite-stream live demo"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, args.size * 2, args.size)

    print(f"\n[priming] B={args.batch_size}, prompt='{args.prompt}'")
    if use_ndi:
        print(f"          source: NDI pattern={args.ndi_source!r} extra_ips={args.ndi_extra_ips!r}")
    print("          first batch incurs compile warmup (30-60s on triton)\n")

    # Source-producer thread: starts reading immediately so cam_q is warm
    # by the time we want to collect batches. cam_q is bounded; if the
    # consumer falls behind, oldest frames are dropped (we want live,
    # not lagged).
    # Small cam_q (batch + 1) so the camera thread can drop-oldest quickly
    # and we never accumulate more than ~one batch worth of stale frames.
    # display_q at maxsize=1 minimises output-side buffering (also using
    # drop-oldest in pipeline_loop's put) — pipeline always shows the most
    # recent finished batch.
    cam_q: Queue = Queue(maxsize=args.batch_size + 1)
    display_q: Queue = Queue(maxsize=1)
    stop_event = threading.Event()
    pause_event = threading.Event()

    if use_ndi:
        cam_thread = threading.Thread(
            target=ndi_camera_loop,
            args=(args.ndi_source, args.ndi_extra_ips, cam_q, args,
                  stop_event, pause_event),
            daemon=True,
        )
    else:
        cam_thread = threading.Thread(
            target=camera_loop,
            args=(cap, cam_q, args, stop_event, pause_event),
            daemon=True,
        )
    cam_thread.start()

    # Prime two batches up front. The first step is where compile warmup
    # happens — keep it on the main thread so the user sees console
    # progress instead of a frozen window. Pull frames from the camera
    # thread's queue so we don't read cap from two threads.
    buf_cur, bgrs_cur = collect_from_cam_q(cam_q, args.batch_size, stop_event)
    if stop_event.is_set() or not buf_cur or len(buf_cur) < args.batch_size:
        print(f"[fatal] source did not deliver a first batch "
              f"(got {len(buf_cur) if buf_cur else 0} frames). "
              f"Check NDI discovery server, sender, and ~/.ndi/ndi-config.v1.json.")
        if cap is not None:
            cap.release()
        return
    pf_cur = worker.prefetch_batch(buf_cur)
    buf_next, bgrs_next = collect_from_cam_q(cam_q, args.batch_size, stop_event)
    pf_next = worker.prefetch_batch(buf_next)
    t0 = time.perf_counter()
    first_outputs, _ = worker.step_batch_with_prefetch(buf_cur, pf_cur)
    t1 = time.perf_counter()
    first_step_dur = t1 - t0
    # First batch wall is dominated by compile warmup; for display pacing we
    # want a sane initial period, not 60 seconds. Use 0.6s as a reasonable
    # placeholder until the pipeline thread reports real iter_dur values.
    first_iter_for_display = min(first_step_dur, 0.6)
    print(f"[ready] first batch took {first_step_dur:.2f}s "
          f"(includes any compile warmup); subsequent batches steady-state\n"
          f"        keys: q quit | space pause | s save | p prompt-change\n")

    primed_state = {
        "buf_cur": buf_cur, "bgrs_cur": bgrs_cur, "pf_cur": pf_cur,
        "buf_next": buf_next, "bgrs_next": bgrs_next, "pf_next": pf_next,
        "first_outputs": first_outputs, "first_batch_dur": first_iter_for_display,
    }
    pipe_thread = threading.Thread(
        target=pipeline_loop,
        args=(worker, args, primed_state, cam_q, display_q,
              stop_event, pause_event),
        daemon=True,
    )
    pipe_thread.start()

    # Live-prompt input thread: reads from stdin without ever pausing
    # the pipeline. Daemon so it dies with the demo on exit.
    prompt_thread = threading.Thread(
        target=prompt_reader_loop,
        args=(args, state, worker, stop_event),
        daemon=True,
    )
    prompt_thread.start()

    save_idx = 0
    fps_smoothed = 0.0  # EMA of pipeline fps
    # Anchor for cross-batch temporal_blend: last output frame we displayed
    # (kept as the *blended* version so seeding into the next batch
    # preserves continuity).
    prev_blended_bgr: "np.ndarray | None" = None
    # Lazy-loaded RIFE model (None until first frame if --interp_method=rife)
    _rife_model = None

    try:
        while not stop_event.is_set():
            try:
                bgrs, outputs, iter_dur = display_q.get(timeout=0.2)
            except Empty:
                k = cv2.waitKey(20) & 0xFF
                if k == ord("q"):
                    stop_event.set()
                continue

            # iter_dur = full pipeline iteration time (collect + prefetch +
            # step). This is the rate at which new batches arrive, so display
            # should pace at the same rate to avoid emptying the queue.
            fps_batch = args.batch_size / max(iter_dur, 1e-6)
            fps_smoothed = (
                0.85 * fps_smoothed + 0.15 * fps_batch
                if fps_smoothed > 0 else fps_batch
            )

            # Convert outputs to BGR up front so we can interpolate uniformly
            # in pixel space.
            out_bgrs = [
                cv2.cvtColor(np.asarray(p), cv2.COLOR_RGB2BGR) for p in outputs
            ]
            in_bgrs = list(bgrs)

            # Display-side temporal alpha blend (applied to outputs only).
            # Smooths static-region flicker; with --temporal_blend_warp the
            # predecessor is Farneback-warped toward the current frame so
            # moving content does not ghost. Seeded with prev_blended_bgr
            # so continuity holds across batch boundaries.
            if args.temporal_blend_alpha > 0.0:
                from dreamlite_stream.frame_interp import temporal_blend
                out_bgrs = temporal_blend(
                    out_bgrs, prev_blended_bgr, args.temporal_blend_alpha,
                    use_flow_warp=args.temporal_blend_warp,
                )
                prev_blended_bgr = out_bgrs[-1] if out_bgrs else prev_blended_bgr

            # Display-side frame interpolation: insert (interp_factor-1)
            # intermediates between each consecutive (in, out) pair.
            # Pipeline throughput is unchanged; this only affects pacing on
            # the display side. The input side always uses linear (camera
            # motion is the ground truth — no need for learned interp);
            # the output side uses RIFE when --interp_method=rife is set,
            # otherwise linear.
            if args.interp_factor > 1:
                from dreamlite_stream.frame_interp import expand_linear
                in_bgrs = expand_linear(in_bgrs, args.interp_factor)
                if args.interp_method == "rife":
                    from dreamlite_stream.frame_interp import (
                        _load_rife, expand_rife,
                    )
                    if _rife_model is None:
                        if not args.rife_path:
                            raise RuntimeError(
                                "--interp_method=rife requires --rife_path "
                                "(directory of a Practical-RIFE checkout). "
                                "See README for setup."
                            )
                        rife_model_arg = args.rife_model or args.rife_path
                        _rife_model = _load_rife(
                            rife_model_arg, args.rife_path, args.device,
                        )
                        print(
                            f"[rife] loaded model from {rife_model_arg} "
                            f"(repo: {args.rife_path})",
                            flush=True,
                        )
                    out_bgrs = expand_rife(
                        out_bgrs, args.interp_factor, _rife_model, args.device,
                    )
                else:
                    out_bgrs = expand_linear(out_bgrs, args.interp_factor)

            # Pace at pipeline rate × interp factor so the expanded sequence
            # fits the same iter_dur window. 1 frame = batch_dur/B/interp sec.
            effective_fps = fps_smoothed * max(1, args.interp_factor)
            per_frame_ms = max(
                5, min(120, int(round(1000.0 / max(effective_fps, 1))))
            )

            for in_bgr, out_bgr in zip(in_bgrs, out_bgrs):
                sxs = np.concatenate([in_bgr, out_bgr], axis=1)
                sxs = overlay_text(sxs, [
                    f"B={args.batch_size}  refresh N={args.cond_refresh_every}  "
                    f"compile={args.compile}",
                    f"fps {fps_smoothed:5.1f}    prompt: {args.prompt[:60]}",
                ])
                cv2.imshow(win_name, sxs)
                k = cv2.waitKey(per_frame_ms) & 0xFF
                if k == ord("q"):
                    stop_event.set()
                    break
                elif k == ord("s"):
                    p = save_dir / f"demo_capture_{save_idx:04d}.png"
                    cv2.imwrite(str(p), out_bgr)
                    print(f"[save] {p}")
                    save_idx += 1
                elif k == ord("p"):
                    print(
                        "[prompt] type a new prompt in the launching terminal "
                        "and press Enter (the pipeline keeps running)",
                        flush=True,
                    )
                elif k == ord(" "):
                    if pause_event.is_set():
                        pause_event.clear()
                        print("[resumed]")
                    else:
                        pause_event.set()
                        print("[paused] press space to resume")
                        # Idle wait for unpause / quit
                        while pause_event.is_set() and not stop_event.is_set():
                            k2 = cv2.waitKey(50) & 0xFF
                            if k2 == ord(" "):
                                pause_event.clear()
                                print("[resumed]")
                            elif k2 == ord("q"):
                                stop_event.set()
                        # Drain stale frames so post-resume display is fresh
                        try:
                            while True:
                                display_q.get_nowait()
                        except Empty:
                            pass
                        break  # leave current display batch
    finally:
        stop_event.set()
        # Drain queues so threads can exit if blocked on put()
        try:
            while True:
                display_q.get_nowait()
        except Empty:
            pass
        try:
            while True:
                cam_q.get_nowait()
        except Empty:
            pass
        pipe_thread.join(timeout=3.0)
        cam_thread.join(timeout=2.0)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        del worker
        gc.collect()
        torch.cuda.empty_cache()
        print("\n[exit] clean")


if __name__ == "__main__":
    main()
