"""CPU-only functional test for dreamlite_stream/rife.py (no RIFE weights needed).

Verifies:

1. The batched forward (N+1 frames, each encoded once) gives the same
   in-between frames as running the network pair by pair.
2. `load_ifnet` reads a Practical-RIFE style checkpoint ("module." prefixed
   keys, extra training-only entries), detects the lite model's narrower
   last block, and rejects a checkpoint that does not fit.
3. `RIFEInterpolator.expand` returns, for each input frame, its in-betweens
   followed by the frame itself; carries the last frame across calls; pads
   and crops frames whose sides are not a multiple of 64; leaves the input
   frames bit-identical; and treats BGR and RGB input consistently.
4. `LinearCarry.expand` produces the same frame count and ordering.

Runs on CPU in a few seconds.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dreamlite_stream import rife  # noqa: E402


def _random_net(c4: int = 32, seed: int = 0) -> rife.IFNet:
    torch.manual_seed(seed)
    net = rife.IFNet(c4=c4).eval()
    with torch.no_grad():
        for p in net.parameters():   # default init gives near-zero flow; make the warps do something
            p.mul_(1.5)
    return net


def test_batched_matches_pairwise():
    net = _random_net()
    torch.manual_seed(1)
    frames = torch.rand(4, 3, 64, 128)
    with torch.no_grad():
        for t in (0.5, 0.25):
            tt = torch.tensor([t])
            batched = net(frames, tt)
            pairwise = torch.cat([net(frames[i:i + 2], tt) for i in range(3)])
            assert batched.shape == (3, 3, 64, 128)
            assert torch.allclose(batched, pairwise, atol=5e-4), (batched - pairwise).abs().max()  # far below 1/255
        # The time position matters, and the output is not just a cross-fade.
        assert not torch.allclose(net(frames, torch.tensor([0.5])), net(frames, torch.tensor([0.25])), atol=1e-4)
    print("[OK] batched forward == pair-by-pair forward")


def test_load_ifnet():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for c4 in (32, 24):
            net = _random_net(c4)
            state = {f"module.{k}": v for k, v in net.state_dict().items()}
            state["module.teacher.conv0.0.0.weight"] = torch.zeros(1)   # training-only entry
            torch.save(state, d / "flownet.pkl")
            loaded = rife.load_ifnet(d)                                   # a folder works too
            assert loaded.block4.conv0[1][0].out_channels == c4
            for k, v in net.state_dict().items():
                assert torch.equal(loaded.state_dict()[k], v), k
        bad = {k: v for k, v in state.items() if "block2" not in k}
        torch.save(bad, d / "flownet.pkl")
        for path, err in ((d / "flownet.pkl", ValueError), (d / "missing.pkl", FileNotFoundError)):
            try:
                rife.load_ifnet(path)
            except err:
                continue
            raise AssertionError(f"expected {err.__name__} for {path.name}")
    print("[OK] load_ifnet: prefixed keys, lite width, bad checkpoints rejected")


def _stub_interpolator() -> rife.RIFEInterpolator:
    """An interpolator whose network is a plain cross-fade, to test the plumbing around it."""
    it = object.__new__(rife.RIFEInterpolator)
    it.device = torch.device("cpu")
    it._trt = it._net = it._prev = None
    it.seen_shapes = []

    def run(seq, t):
        it.seen_shapes.append(tuple(seq.shape))
        return (1 - t) * seq[:-1].float() + t * seq[1:].float()

    it._run = run
    return it


def _frames(values, h=64, w=64):
    """Frames of constant colour: values is a list of (r, g, b)."""
    return np.stack([np.broadcast_to(np.array(v, np.uint8), (h, w, 3)).copy() for v in values])


def test_expand_order_and_carry():
    it = _stub_interpolator()
    a = _frames([(0, 40, 80), (100, 140, 180), (200, 240, 20)])
    out = it.expand(a, 2)
    assert out.shape == (6, 64, 64, 3) and out.dtype == np.uint8
    assert np.array_equal(out[1::2], a)                       # originals, untouched
    assert np.array_equal(out[0], a[0])                       # first call: no earlier frame
    assert np.array_equal(out[2, 0, 0], [50, 90, 130])        # between frame 0 and 1
    assert np.array_equal(out[4, 0, 0], [150, 190, 100])
    b = _frames([(0, 0, 0)])
    out2 = it.expand(b, 2)
    assert out2.shape == (2, 64, 64, 3)
    assert np.array_equal(out2[0, 0, 0], [100, 120, 10])      # across the batch boundary
    it.reset()
    assert np.array_equal(it.expand(b, 2)[0], b[0])           # reset drops the carried frame
    assert np.array_equal(it.expand(list(a), 1), a)           # factor 1: unchanged, lists accepted
    print("[OK] expand: in-betweens then the frame, last frame carried across calls")


def test_expand_factor_4():
    it = _stub_interpolator()
    a = _frames([(0, 0, 0), (200, 100, 40)])
    out = it.expand(a, 4)
    assert out.shape == (8, 64, 64, 3)
    assert [tuple(px) for px in out[4:8, 0, 0]] == [(50, 25, 10), (100, 50, 20), (150, 75, 30), (200, 100, 40)]
    print("[OK] expand: factor 4 places three in-betweens in time order")


def test_expand_padding_and_bgr():
    it = _stub_interpolator()
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(3, 40, 72, 3), dtype=np.uint8)
    out = it.expand(a, 2)
    assert out.shape == (6, 40, 72, 3)
    assert it.seen_shapes[-1] == (4, 3, 64, 128)              # padded up to multiples of 64
    assert np.array_equal(out[1::2], a)
    want = np.round((a[0].astype(np.float64) + a[1]) / 2)
    assert np.abs(out[2].astype(np.float64) - want).max() <= 1
    # BGR in, BGR out: same result as converting by hand around an RGB call.
    it_rgb, it_bgr = _stub_interpolator(), _stub_interpolator()
    assert np.array_equal(it_bgr.expand(a, 2, bgr=True), it_rgb.expand(a[..., ::-1], 2)[..., ::-1])
    print("[OK] expand: pads to a multiple of 64 and crops back; BGR handled")


def test_linear_carry():
    lc = rife.LinearCarry()
    a = list(_frames([(0, 0, 0), (100, 100, 100)]))
    out = lc.expand(a, 2)
    assert len(out) == 4
    assert out[0][0, 0, 0] == 0 and out[1][0, 0, 0] == 0 and out[2][0, 0, 0] == 50 and out[3][0, 0, 0] == 100
    out2 = lc.expand(list(_frames([(200, 200, 200)])), 2)
    assert len(out2) == 2 and out2[0][0, 0, 0] == 150
    assert len(rife.LinearCarry().expand(a, 4)) == 8
    print("[OK] LinearCarry: same count and ordering as the RIFE path")


if __name__ == "__main__":
    test_batched_matches_pairwise()
    test_load_ifnet()
    test_expand_order_and_carry()
    test_expand_factor_4()
    test_expand_padding_and_bgr()
    test_linear_carry()
    print("\nALL RIFE INTERPOLATION TESTS PASSED")
