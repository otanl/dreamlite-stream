"""CPU-only functional test for dreamlite_stream.onestep_lora.

Verifies, on a tiny stand-in module tree with the UNet's layer naming:

1. `attach` wraps exactly the layers of the requested groups (attn / ff / pw)
   and never a depthwise convolution; a fresh wrapper changes nothing.
2. Merging the deltas into the weights reproduces the LoRA-branch forward,
   for Linear and for 1x1 Conv2d layers, and honours `scale`.
3. `restore()` puts the original weights back bit-for-bit.
4. Merging into bfloat16 weights rounds exactly once.
5. A layer whose `forward` was patched (as LLLite does) is still merged and
   keeps its patched forward.
6. save / load round-trips tensors and typed metadata.
7. `compatibility_notes` flags the mismatches that matter at run time.

Runs on CPU in a few seconds; no model weights needed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dreamlite_stream import onestep_lora as ol  # noqa: E402


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.to_q, self.to_k, self.to_v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.to_out = nn.ModuleList([nn.Linear(d, d), nn.Dropout(0.0)])

    def forward(self, x):
        return self.to_out[0](self.to_q(x) + self.to_k(x) + self.to_v(x))


class _Proj(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, 2 * d)

    def forward(self, x):
        return torch.nn.functional.gelu(self.proj(x))


class _FF(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.ModuleList([_Proj(d), nn.Dropout(0.0), nn.Linear(2 * d, d)])

    def forward(self, x):
        for m in self.net:
            x = m(x)
        return x


class _TBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn1, self.attn2, self.ff = _Attn(d), _Attn(d), _FF(d)

    def forward(self, x):
        return self.ff(self.attn2(self.attn1(x)))


class _SepConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.depthwise = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.pointwise = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class _Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1, self.conv2 = _SepConv(c), _SepConv(c)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class _Holder(nn.Module):
    pass


def _fake_unet(d=8):
    torch.manual_seed(0)
    u = _Holder()
    blk = _Holder()
    att = _Holder()
    att.transformer_blocks = nn.ModuleList([_TBlock(d)])
    blk.attentions = nn.ModuleList([att])
    blk.resnets = nn.ModuleList([_Res(d)])
    u.down_blocks = nn.ModuleList([blk])
    return u


def _run(u, x_tok, x_img):
    tb = u.down_blocks[0].attentions[0].transformer_blocks[0]
    return tb(x_tok), u.down_blocks[0].resnets[0](x_img)


def _randomise(wrappers):
    for w in wrappers.values():
        nn.init.normal_(w.B, std=0.1)


def test_attach_targets():
    for targets, n in ((("attn",), 8), (("ff",), 2), (("pw",), 2), (("attn", "ff", "pw"), 12)):
        u = _fake_unet()
        w = ol.attach(u, rank=2, targets=targets)
        assert len(w) == n, (targets, len(w))
        assert not any(name.endswith("depthwise") for name in w)
        ol.detach(u, w)
        assert not any(isinstance(m, ol.LoRAWrapper) for m in u.modules())
    try:
        ol.attach(_fake_unet(), rank=2, targets=("nope",))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown target group must raise")
    print("[OK] attach wraps exactly the requested groups; depthwise convs untouched; detach undoes it")


def test_fresh_wrapper_is_identity():
    u = _fake_unet()
    x_tok, x_img = torch.randn(2, 5, 8), torch.randn(2, 8, 6, 6)
    base = _run(u, x_tok, x_img)
    ol.attach(u, rank=2, targets=("attn", "ff", "pw"))
    out = _run(u, x_tok, x_img)
    assert all(torch.equal(a, b) for a, b in zip(base, out))
    print("[OK] a fresh LoRA (B = 0) leaves the output unchanged")


def test_merge_matches_branch_and_restore_is_exact():
    u = _fake_unet()
    x_tok, x_img = torch.randn(2, 5, 8), torch.randn(2, 8, 6, 6)
    base = _run(u, x_tok, x_img)
    original = {k: v.clone() for k, v in u.state_dict().items()}
    w = ol.attach(u, rank=3, alpha=6.0, targets=("attn", "ff", "pw"))
    _randomise(w)
    branch = _run(u, x_tok, x_img)
    assert not torch.allclose(branch[0], base[0]) and not torch.allclose(branch[1], base[1])
    state = ol.state_from_wrappers(w)
    meta = dict(rank=3, alpha=6.0)
    ol.detach(u, w)

    handle = ol.merge(u, state, meta)
    merged = _run(u, x_tok, x_img)
    assert handle.n_layers == 12
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(branch, merged))

    handle.restore()
    assert all(torch.equal(v, original[k]) for k, v in u.state_dict().items())
    assert all(torch.equal(a, b) for a, b in zip(base, _run(u, x_tok, x_img)))

    half = ol.merge(u, state, meta, scale=0.5)
    name = ol.layer_names(state)[0]
    got = u.get_submodule(name).weight - original[name + ".weight"]
    want = ol.delta_weight(state, name, 6.0 / 3 * 0.5, got)
    assert torch.allclose(got, want, atol=1e-6)
    half.restore()
    print("[OK] merged weights reproduce the LoRA branch (Linear and 1x1 conv); scale honoured; restore exact")


def test_merge_into_bf16_rounds_once():
    u = _fake_unet().to(torch.bfloat16)
    w = ol.attach(u, rank=2, targets=("attn",))
    _randomise(w)
    state = ol.state_from_wrappers(w)
    ol.detach(u, w)
    name = ol.layer_names(state)[0]
    before = u.get_submodule(name).weight.detach().clone()
    ol.merge(u, state, dict(rank=2, alpha=2.0))
    after = u.get_submodule(name).weight
    want = (before.float() + ol.delta_weight(state, name, 1.0, before)).to(torch.bfloat16)
    assert after.dtype == torch.bfloat16 and torch.equal(after, want)
    print("[OK] merging into bfloat16 weights = fp32 sum rounded once")


def test_merge_keeps_patched_forward():
    u = _fake_unet()
    lin = u.down_blocks[0].attentions[0].transformer_blocks[0].attn1.to_q
    calls = []
    orig_forward = lin.forward

    def patched(x):  # what LLLite's inference hook does: add something, then call the original
        calls.append(1)
        return orig_forward(x + 0.0)

    lin.forward = patched
    w0 = lin.weight.detach().clone()
    state = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A": torch.randn(2, 8),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B": torch.randn(8, 2),
    }
    ol.merge(u, state, dict(rank=2, alpha=2.0))
    assert not torch.equal(lin.weight, w0)
    _run(u, torch.randn(1, 3, 8), torch.randn(1, 8, 4, 4))
    assert calls, "patched forward must still be the one that runs"
    print("[OK] a layer with a patched forward is merged and keeps its patch")


def test_save_load_roundtrip():
    u = _fake_unet()
    w = ol.attach(u, rank=2, targets=("attn", "pw"))
    _randomise(w)
    state = ol.state_from_wrappers(w)
    n_layers = len(ol.layer_names(state))
    assert ol.init_noise(state) is None
    noise = torch.randn(1, 4, 8, 8)
    state[ol.NOISE_KEY] = noise
    meta = dict(size=768, seed=42, rank=2, alpha=2.0, targets=["attn", "pw"], prompts=["a", "b c"])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.safetensors"
        ol.save(path, state, meta)
        state2, meta2 = ol.load(path)
        assert torch.equal(ol.load_init_noise(path), noise)
    assert meta2 == meta
    assert set(state2) == set(state) and all(torch.equal(state[k], state2[k]) for k in state)
    # The stored init noise travels with the file and is not mistaken for a layer.
    assert torch.equal(ol.init_noise(state2), noise)
    assert len(ol.layer_names(state2)) == n_layers
    ol.detach(u, w)
    assert ol.merge(u, state2, meta2).n_layers == n_layers
    print("[OK] save / load round-trips tensors, the init noise and typed metadata")


def test_compatibility_notes():
    meta = dict(size=768, seed=42, prompts=["p1"])
    run = dict(size=768, steps=1, shared_noise=True, prompt="p1")
    # A file that carries its init noise: matched is silent, and the seed is irrelevant.
    assert ol.compatibility_notes(meta, seed=42, has_noise=True, **run) == []
    assert ol.compatibility_notes(meta, seed=7, has_noise=True, **run) == []
    # A file without it depends on the seed reproducing the tensor, and says so.
    assert len(ol.compatibility_notes(meta, seed=42, **run)) == 1
    assert len(ol.compatibility_notes(meta, seed=7, **run)) == 2
    bad = ol.compatibility_notes(meta, size=640, seed=7, steps=2, shared_noise=False, prompt="other", has_noise=True)
    assert len(bad) == 4
    print("[OK] compatibility_notes: silent when matched, one note per mismatch, seed ignored with stored noise")


if __name__ == "__main__":
    test_attach_targets()
    test_fresh_wrapper_is_identity()
    test_merge_matches_branch_and_restore_is_exact()
    test_merge_into_bf16_rounds_once()
    test_merge_keeps_patched_forward()
    test_save_load_roundtrip()
    test_compatibility_notes()
    print("\nALL ONE-STEP LORA TESTS PASSED")
