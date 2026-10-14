"""CPU-only functional test for the lean-path options of BatchedEditWorker.

Verifies, without loading any model weights:

1. `--lean` resolves to the documented preset, explicit refresh values win,
   and `--vae_compile` implies `--fast_vae`.
2. Init-noise policies: the default draws new noise per batch, `fixed_noise`
   repeats one (B, C, h, w) tensor whose slots differ, and `shared_noise`
   gives every frame of every batch the same pattern.
3. Refresh schedules: `te_refresh_every` / `cond_refresh_every` fire as
   before for N >= 1, and 0 means "first call only".
4. `reset_sequence()` returns the worker to its first-call behaviour.
5. The uint8 I/O helpers normalise and quantise exactly, and accept PIL
   images and arrays interchangeably.
6. `BatchTiming.loop_ms` adds the prefetch time to the step time.
7. `--trt_engine`: the engine's spec supplies the adapter multiplier when
   none is given, and a spec reports every way it does not fit a run.

Runs on CPU in a few seconds; no GPU needed.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dreamlite_stream import BatchedEditWorker, SharedState  # noqa: E402
from dreamlite_stream import pipeline_ops as ops  # noqa: E402
from dreamlite_stream.lean import (  # noqa: E402
    add_lean_args, holds_clip_state, lean_worker_kwargs, resolve_lean_args,
)
from dreamlite_stream.trt_engine import EngineSpec  # noqa: E402
from dreamlite_stream.workers.batched_edit import BatchTiming  # noqa: E402


def _stub_pipeline():
    """Just enough of a pipeline for the noise and decode helpers."""
    vae_config = SimpleNamespace(latent_channels=4, scaling_factor=1.0, shift_factor=0.0)
    return SimpleNamespace(vae_scale_factor=8, vae=SimpleNamespace(config=vae_config))


def _worker(**kw) -> BatchedEditWorker:
    state = SharedState(height=64, width=64, num_inference_steps=1, prompt="p")
    return BatchedEditWorker(
        pipeline=_stub_pipeline(), state=state, batch_size=4, device="cpu",
        dtype=torch.float32, seed=42, compile=False, cond_flow_workers=0, **kw,
    )


def _parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--cond_refresh_every", type=int, default=None)
    add_lean_args(p)
    return resolve_lean_args(p.parse_args(argv))


def test_lean_preset_resolution():
    a = _parse([])
    assert (a.cond_refresh_every, a.te_refresh_every) == (8, 1)
    assert not (a.te_batch_one or a.shared_noise or a.fast_vae or a.vae_compile)
    assert a.onestep_lora is None and a.lllite_multiplier == 1.0
    assert lean_worker_kwargs(a)["output_type"] == "pil"
    assert not holds_clip_state(a)

    a = _parse(["--lean"])
    assert (a.cond_refresh_every, a.te_refresh_every) == (0, 0)
    assert a.te_batch_one and a.shared_noise and a.fast_vae and a.vae_compile
    assert a.onestep_lora is None  # a LoRA is never implied; it is tied to a size, seed and prompt set
    assert lean_worker_kwargs(a)["output_type"] == "np"
    assert holds_clip_state(a)

    a = _parse(["--lean", "--cond_refresh_every", "4", "--te_refresh_every", "16"])
    assert (a.cond_refresh_every, a.te_refresh_every) == (4, 16)

    a = _parse(["--vae_compile"])
    assert a.fast_vae and not a.shared_noise
    print("[OK] --lean resolves to the preset; explicit values win; vae_compile implies fast_vae")


def test_default_noise_changes_per_batch():
    w = _worker()
    a = w._make_init_latents(4)
    w.state.frame_idx += 4
    b = w._make_init_latents(4)
    assert a.shape == (4, 4, 8, 8)
    assert not torch.equal(a, b)
    assert not torch.equal(a[0], a[1])
    print("[OK] default: new noise every batch, slots differ")


def test_fixed_noise_is_per_slot():
    w = _worker(fixed_noise=True)
    a = w._make_init_latents(4)
    w.state.frame_idx += 4
    b = w._make_init_latents(4)
    assert torch.equal(a, b)
    assert not torch.equal(a[0], a[1])
    print("[OK] fixed_noise: same tensor every batch, but slots still differ")


def test_shared_noise_is_one_pattern():
    w = _worker(shared_noise=True)
    a = w._make_init_latents(4)
    w.state.frame_idx += 4
    b = w._make_init_latents(4)
    assert a.shape == (4, 4, 8, 8) and a.is_contiguous()
    assert all(torch.equal(a[0], a[i]) for i in range(4))
    assert torch.equal(a, b)
    assert a.data_ptr() != b.data_ptr()  # fresh storage per call for the compiled UNet
    # Matches the single-frame draw the default path would make at frame 0.
    ref = ops.make_init_noise(
        w.pipeline, height=64, width=64, device="cpu", dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(42), batch_size=1,
    )
    assert torch.equal(a[:1], ref)
    # Without a seed the pattern is still drawn once and then held.
    w2 = _worker(shared_noise=True)
    w2.seed = None
    assert torch.equal(w2._make_init_latents(4), w2._make_init_latents(4))
    # A resolution change redraws at the new latent size.
    w.state.height = w.state.width = 128
    assert w._make_init_latents(2).shape == (2, 4, 16, 16)
    print("[OK] shared_noise: one pattern for every frame of every batch")


def _te_ticks(w, n, cached_after_first=True):
    out = []
    for _ in range(n):
        due = w._te_refresh_due()
        out.append(due)
        if due and cached_after_first:
            w._cached_te_embeds = torch.zeros(1)  # what a refresh leaves behind
    return out


def test_te_refresh_schedule():
    assert _te_ticks(_worker(te_refresh_every=1), 4) == [True, True, True, True]
    assert _te_ticks(_worker(te_refresh_every=4), 9) == [True, False, False, False, True, False, False, False, True]
    w = _worker(te_refresh_every=0)
    assert _te_ticks(w, 5) == [True, False, False, False, False]
    w._force_te_refresh = True  # what a prompt change sets
    assert _te_ticks(w, 3) == [True, False, False]
    print("[OK] te_refresh_every: N>=1 unchanged, 0 = first call and forced refresh only")


def _cond_ticks(w, n):
    out = []
    for _ in range(n):
        out.append(w._cond_refresh_due())
        w._cond_call_idx += 1  # what _set_lllite_cond_for_batch does
    return out


def test_cond_refresh_schedule():
    assert _cond_ticks(_worker(cond_refresh_every=1), 3) == [True, True, True]
    assert _cond_ticks(_worker(cond_refresh_every=8), 9) == [True] + [False] * 7 + [True]
    assert _cond_ticks(_worker(cond_refresh_every=0), 20) == [True] + [False] * 19
    print("[OK] cond_refresh_every: N>=1 unchanged, 0 = first call only")


def test_reset_sequence():
    w = _worker(te_refresh_every=0, cond_refresh_every=0)
    _te_ticks(w, 3)
    _cond_ticks(w, 3)
    w.state.frame_idx = 24
    w._last_prev_decoded = np.zeros((2, 2, 3), np.uint8)
    w._last_prev_input_gray = np.zeros((2, 2), np.uint8)
    w.reset_sequence()
    assert w.state.frame_idx == 0
    assert w._last_prev_decoded is None and w._last_prev_input_gray is None
    assert w._cached_te_embeds is None
    assert w._te_refresh_due() is True
    assert w._cond_refresh_due() is True
    print("[OK] reset_sequence restores first-call behaviour")


def test_option_validation():
    for bad in (dict(output_type="tensor"), dict(vae_compile=True)):
        try:
            _worker(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
    print("[OK] bad output_type and vae_compile-without-fast_vae are rejected")


def test_uint8_io_helpers():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    assert np.array_equal(ops.as_rgb_array(pil), arr)
    assert ops.as_rgb_array(arr) is arr
    assert np.array_equal(np.asarray(ops.as_pil(arr)), arr)
    assert ops.as_pil(pil) is pil

    # PIL and array frames mix freely; a frame at the wrong size is resized.
    small = Image.fromarray(rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8))
    batch = ops.frames_to_uint8_batch([pil, arr, small], height=16, width=16)
    assert batch.shape == (3, 16, 16, 3) and batch.dtype == torch.uint8
    assert np.array_equal(batch[0].numpy(), arr) and np.array_equal(batch[1].numpy(), arr)

    # Normalisation: 0 -> -1, 255 -> +1, NCHW out. Identity "VAE" exposes it.
    edge = torch.tensor([[[[0, 255, 51]]]], dtype=torch.uint8)  # (1, 1, 1, 3)
    x = ops.encode_uint8_batch_to_latent(lambda t: t, edge, "cpu", torch.float32, torch.float32)
    assert x.shape == (1, 3, 1, 1)
    assert torch.allclose(x.flatten(), torch.tensor([-1.0, 1.0, -0.6]), atol=1e-6)

    # Decode quantises on the tensor side and round-trips an encode exactly.
    full = torch.from_numpy(arr)[None]
    lat = ops.encode_uint8_batch_to_latent(lambda t: t, full, "cpu", torch.float32, torch.float32)
    back = ops.decode_latent_to_uint8(_stub_pipeline(), lambda t: t, lat, torch.float32)
    assert back.shape == (1, 16, 16, 3) and back.dtype == np.uint8
    assert np.array_equal(back[0], arr)
    # Out-of-range decoder output is clamped, not wrapped.
    wild = torch.tensor([-3.0, 3.0, 0.0]).view(1, 3, 1, 1)
    assert ops.decode_latent_to_uint8(_stub_pipeline(), lambda t: t, wild, torch.float32).flatten().tolist() == [0, 255, 128]
    print("[OK] uint8 I/O helpers: exact normalise / quantise, PIL and arrays interchangeable")


def test_loop_ms():
    t = BatchTiming(total_ms=200.0, prefetch_ms=250.0, n_frames=8)
    assert t.loop_ms == 450.0
    assert t.per_frame_ms == 25.0  # step-only, unchanged
    print("[OK] BatchTiming.loop_ms = step + prefetch")


def test_trt_engine_spec():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        lora = d / "a.safetensors"
        lora.write_bytes(b"weights A")
        other = d / "b.safetensors"
        other.write_bytes(b"weights B")
        from dreamlite_stream.trt_engine import file_sha256

        # An engine with a LoRA and no adapter.
        plain = d / "plain.engine"
        EngineSpec(size=768, batch_size=8, seq_len=192, tensorrt="10.16.1",
                   onestep_lora=lora.name, onestep_lora_sha256=file_sha256(lora)).save(plain)
        spec = EngineSpec.load(plain)
        assert not spec.has_lllite
        assert spec.mismatches(size=768, batch_size=8, lllite_multiplier=0.0) == []
        assert spec.mismatches(size=768, batch_size=8, onestep_lora=str(lora), lllite_multiplier=0.0) == []
        bad = spec.mismatches(size=640, batch_size=4, onestep_lora=str(other), lllite_multiplier=1.0)
        assert len(bad) == 4, bad
        # With no explicit multiplier the engine decides; an explicit one is kept.
        assert _parse(["--lean", "--trt_engine", str(plain)]).lllite_multiplier == 0.0
        assert _parse(["--lean", "--trt_engine", str(plain), "--lllite_multiplier", "1"]).lllite_multiplier == 1.0

        # An engine with the adapter built in at half strength.
        adapted = d / "adapted.engine"
        EngineSpec(size=640, batch_size=4, seq_len=192, lllite_weights="w.safetensors",
                   lllite_blocks="down_blocks", lllite_multiplier=0.5, hook_names=["h0", "h1"]).save(adapted)
        spec = EngineSpec.load(adapted)
        assert spec.has_lllite
        assert _parse(["--trt_engine", str(adapted)]).lllite_multiplier == 0.5
        assert spec.mismatches(size=640, batch_size=4, lllite_multiplier=0.5) == []
        assert len(spec.mismatches(size=640, batch_size=4, lllite_multiplier=1.0)) == 1
        # A LoRA asked for on the command line that the engine does not contain.
        assert len(spec.mismatches(size=640, batch_size=4, onestep_lora=str(lora), lllite_multiplier=0.5)) == 1
        # An engine built on another platform does not load here; one with no platform recorded is not judged.
        assert spec.platform == ""
        foreign = d / "foreign.engine"
        EngineSpec(size=640, batch_size=4, seq_len=192, platform="some-other-os").save(foreign)
        assert len(EngineSpec.load(foreign).mismatches(size=640, batch_size=4, lllite_multiplier=0.0)) == 1
        here = d / "here.engine"
        EngineSpec(size=640, batch_size=4, seq_len=192, platform=sys.platform).save(here)
        assert EngineSpec.load(here).mismatches(size=640, batch_size=4, lllite_multiplier=0.0) == []

        try:
            EngineSpec.load(d / "missing.engine")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError for an engine without a spec")
    print("[OK] --trt_engine: the spec supplies the adapter multiplier and reports mismatches")


if __name__ == "__main__":
    test_lean_preset_resolution()
    test_default_noise_changes_per_batch()
    test_fixed_noise_is_per_slot()
    test_shared_noise_is_one_pattern()
    test_te_refresh_schedule()
    test_cond_refresh_schedule()
    test_reset_sequence()
    test_option_validation()
    test_uint8_io_helpers()
    test_loop_ms()
    test_trt_engine_spec()
    print("\nALL LEAN OPTION TESTS PASSED")
