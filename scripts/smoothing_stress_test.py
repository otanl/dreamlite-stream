"""Smoothing-collapse metric stress test (paper, sec.3.4 + appendix C).

Validates that our two smoothing metrics -- Sobel mean-abs and HF-FFT
log-ratio -- have a *monotone* response under (a) Gaussian blur and
(b) unsharp mask, applied to a real image batch from DAVIS-2017.

This is a paper-grade evidence script: it produces a CSV table and a
matplotlib figure that go into the paper as the §3.4 metric
validation.

Usage::

    python scripts/smoothing_stress_test.py \\
        --davis-root assets/davis/DAVIS/JPEGImages/480p \\
        --num-frames 32 \\
        --resolution 512 \\
        --out-dir out/smoothing_stress

Output files (under `out/smoothing_stress/`):
    - blur_sweep.csv          Sobel + HF-FFT readings at each sigma
    - sharpen_sweep.csv       same at each unsharp k
    - stress_test.png         two-panel plot (blur left, sharpen right)
    - stress_test.pdf         same plot as PDF (for LaTeX inclusion)

Runtime on RTX 3090 Ti: ~30 seconds for 32 frames at 512^2.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def sobel_mean_abs(images: torch.Tensor) -> torch.Tensor:
    """Per-image mean absolute Sobel response, averaged over RGB.

    Args:
        images: (N, 3, H, W) in [0, 1] or [-1, 1] -- direction-only metric.

    Returns:
        (N,) tensor of mean |Sobel| per image.
    """
    N, C, H, W = images.shape
    assert C == 3
    # Sobel-X and Sobel-Y kernels broadcast to all 3 channels.
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    flat = images.view(N * C, 1, H, W)
    gx = F.conv2d(flat, kx, padding=1)
    gy = F.conv2d(flat, ky, padding=1)
    mag = (gx * gx + gy * gy).sqrt()
    return mag.view(N, C, H, W).mean(dim=(1, 2, 3))


def _hf_mask(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Boolean mask of FFT bins outside the center disc of radius min(h,w)/4.

    Matches the mask used by dreamlite_stream.distillation_losses for the
    HF-FFT spectral loss, so this stress test directly validates the
    metric that the loss is built on top of.
    """
    cy, cx = h // 2, w // 2
    y = torch.arange(h, device=device).view(-1, 1)
    x = torch.arange(w, device=device).view(1, -1)
    r2 = (y - cy) ** 2 + (x - cx) ** 2
    return r2 > (min(h, w) // 4) ** 2


def hf_fft_log_ratio(target: torch.Tensor, reference: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
    """log(target HF energy / reference HF energy), per image.

    Both inputs share the same shape (N, C, H, W). Grayscale is taken by
    channel-mean before FFT; HF mask is shared across the batch.

    Returns (N,) tensor in [-inf, +inf]. Zero == identical HF energy.
    Negative == target has *less* HF energy (smoothing collapse).
    Positive == target has *more* HF energy (over-sharpening).
    """
    assert target.shape == reference.shape
    s = target.float().mean(dim=1)         # (N, H, W)
    t = reference.float().mean(dim=1)      # (N, H, W)
    s_fft = torch.fft.fftshift(torch.fft.fft2(s), dim=(-2, -1)).abs()
    t_fft = torch.fft.fftshift(torch.fft.fft2(t), dim=(-2, -1)).abs()
    _, h, w = s_fft.shape
    mask = _hf_mask(h, w, s.device)
    s_hf = s_fft[:, mask].mean(dim=-1)
    t_hf = t_fft[:, mask].mean(dim=-1)
    return torch.log(s_hf + eps) - torch.log(t_hf + eps)


# ---------------------------------------------------------------------------
# Treatments (deterministic, in-place per batch)
# ---------------------------------------------------------------------------


def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """2D separable Gaussian blur with reflection padding.

    sigma == 0 returns a clone (identity).
    """
    if sigma <= 0:
        return images.clone()
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=images.dtype,
                     device=images.device)
    k = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    kx = k.view(1, 1, 1, -1).expand(images.size(1), 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(images.size(1), 1, -1, 1)
    pad = radius
    x_padded = F.pad(images, (pad, pad, 0, 0), mode="reflect")
    x_blur_h = F.conv2d(x_padded, kx, groups=images.size(1))
    x_padded = F.pad(x_blur_h, (0, 0, pad, pad), mode="reflect")
    x_blur = F.conv2d(x_padded, ky, groups=images.size(1))
    return x_blur


def unsharp_mask(images: torch.Tensor, k: float, sigma: float = 1.0
                 ) -> torch.Tensor:
    """Standard unsharp mask: out = orig + k * (orig - blur(orig, sigma)).

    k == 0 returns a clone (identity).
    """
    if k <= 0:
        return images.clone()
    blurred = gaussian_blur(images, sigma)
    return torch.clamp(images + k * (images - blurred), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_davis_batch(root: Path, num_frames: int, resolution: int,
                     seed: int = 0) -> torch.Tensor:
    """Load num_frames frames from random sequences/positions in DAVIS.

    Returns a (N, 3, R, R) tensor in [0, 1] on CPU.
    """
    if not root.is_dir():
        sys.exit(f"DAVIS root not found: {root}")
    sequences = sorted(p for p in root.iterdir() if p.is_dir())
    if not sequences:
        sys.exit(f"No sequences in {root}")
    rng = random.Random(seed)
    frames = []
    while len(frames) < num_frames:
        seq = rng.choice(sequences)
        jpgs = sorted(seq.glob("*.jpg"))
        if not jpgs:
            continue
        jpg = rng.choice(jpgs)
        im = Image.open(jpg).convert("RGB")
        # Center-crop to square then resize.
        w, h = im.size
        s = min(w, h)
        left, upper = (w - s) // 2, (h - s) // 2
        im = im.crop((left, upper, left + s, upper + s))
        im = im.resize((resolution, resolution), Image.BICUBIC)
        arr = torch.tensor(list(im.getdata()), dtype=torch.uint8)
        arr = arr.view(resolution, resolution, 3).permute(2, 0, 1).float() / 255.0
        frames.append(arr)
    return torch.stack(frames, dim=0)


# ---------------------------------------------------------------------------
# Sweep + plotting
# ---------------------------------------------------------------------------


@dataclass
class SweepRow:
    treatment: str
    parameter: float
    sobel_mean: float
    sobel_std: float
    hf_log_ratio_mean: float
    hf_log_ratio_std: float


def run_sweep(batch: torch.Tensor, treatment_fn, values, ref_batch,
              treatment_name: str) -> list[SweepRow]:
    rows = []
    for v in values:
        treated = treatment_fn(batch, v)
        sobel = sobel_mean_abs(treated)
        hfr = hf_fft_log_ratio(treated, ref_batch)
        rows.append(SweepRow(
            treatment=treatment_name,
            parameter=float(v),
            sobel_mean=float(sobel.mean().item()),
            sobel_std=float(sobel.std().item()),
            hf_log_ratio_mean=float(hfr.mean().item()),
            hf_log_ratio_std=float(hfr.std().item()),
        ))
    return rows


def write_csv(path: Path, rows: list[SweepRow]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["treatment", "parameter",
                    "sobel_mean", "sobel_std",
                    "hf_log_ratio_mean", "hf_log_ratio_std"])
        for r in rows:
            w.writerow([r.treatment, r.parameter,
                        r.sobel_mean, r.sobel_std,
                        r.hf_log_ratio_mean, r.hf_log_ratio_std])


def plot_results(blur_rows: list[SweepRow], sharp_rows: list[SweepRow],
                 png_path: Path, pdf_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0), constrained_layout=True)

    # Blur: Sobel
    xs = [r.parameter for r in blur_rows]
    ax = axes[0, 0]
    ax.errorbar(xs, [r.sobel_mean for r in blur_rows],
                yerr=[r.sobel_std for r in blur_rows],
                marker="o", capsize=3)
    ax.set_xlabel(r"Gaussian blur $\sigma$")
    ax.set_ylabel("Sobel mean-abs")
    ax.set_title("(a) Sobel under blur (should decrease)")
    ax.grid(True, alpha=0.3)

    # Blur: HF-FFT log-ratio
    ax = axes[0, 1]
    ax.errorbar(xs, [r.hf_log_ratio_mean for r in blur_rows],
                yerr=[r.hf_log_ratio_std for r in blur_rows],
                marker="o", capsize=3, color="C1")
    ax.axhline(0.0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"Gaussian blur $\sigma$")
    ax.set_ylabel(r"$\log(\text{treated HF} / \text{reference HF})$")
    ax.set_title("(b) HF-FFT log-ratio under blur (should decrease)")
    ax.grid(True, alpha=0.3)

    # Sharpen: Sobel
    xs = [r.parameter for r in sharp_rows]
    ax = axes[1, 0]
    ax.errorbar(xs, [r.sobel_mean for r in sharp_rows],
                yerr=[r.sobel_std for r in sharp_rows],
                marker="s", capsize=3, color="C2")
    ax.set_xlabel(r"Unsharp-mask $k$")
    ax.set_ylabel("Sobel mean-abs")
    ax.set_title("(c) Sobel under unsharp (should increase)")
    ax.grid(True, alpha=0.3)

    # Sharpen: HF-FFT log-ratio
    ax = axes[1, 1]
    ax.errorbar(xs, [r.hf_log_ratio_mean for r in sharp_rows],
                yerr=[r.hf_log_ratio_std for r in sharp_rows],
                marker="s", capsize=3, color="C3")
    ax.axhline(0.0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"Unsharp-mask $k$")
    ax.set_ylabel(r"$\log(\text{treated HF} / \text{reference HF})$")
    ax.set_title("(d) HF-FFT log-ratio under unsharp (should increase)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Smoothing-collapse metric stress test (§3.4)",
                 fontsize=11)
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis-root", type=Path,
                    default=Path("assets/davis/DAVIS/JPEGImages/480p"))
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=Path("out/smoothing_stress"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}, frames={args.num_frames}, res={args.resolution}")

    print("loading DAVIS batch ...")
    batch_cpu = load_davis_batch(args.davis_root, args.num_frames,
                                 args.resolution, seed=args.seed)
    batch = batch_cpu.to(device)
    print(f"batch shape {tuple(batch.shape)}, range "
          f"[{batch.min():.3f}, {batch.max():.3f}]")

    # Reference (sigma=0 / k=0) for HF-FFT log-ratio is the same untouched batch.
    blur_sigmas = [0.0, 0.5, 1.0, 2.0, 4.0]
    sharp_ks = [0.0, 0.5, 1.0, 2.0]
    print(f"blur sweep sigma in {blur_sigmas}")
    print(f"sharpen sweep k in {sharp_ks}")

    blur_rows = run_sweep(batch, gaussian_blur, blur_sigmas, batch, "blur")
    sharp_rows = run_sweep(batch, unsharp_mask, sharp_ks, batch, "unsharp")

    blur_csv = args.out_dir / "blur_sweep.csv"
    sharp_csv = args.out_dir / "sharpen_sweep.csv"
    write_csv(blur_csv, blur_rows)
    write_csv(sharp_csv, sharp_rows)
    print(f"wrote {blur_csv}")
    print(f"wrote {sharp_csv}")

    plot_results(blur_rows, sharp_rows,
                 args.out_dir / "stress_test.png",
                 args.out_dir / "stress_test.pdf")

    # Print summary as text for paper inclusion.
    print("\n=== blur sweep ===")
    for r in blur_rows:
        print(f"  sigma={r.parameter:.2f}  Sobel={r.sobel_mean:.4f}+/-{r.sobel_std:.4f}  "
              f"HFlog={r.hf_log_ratio_mean:+.4f}+/-{r.hf_log_ratio_std:.4f}")
    print("\n=== unsharp sweep ===")
    for r in sharp_rows:
        print(f"  k={r.parameter:.2f}  Sobel={r.sobel_mean:.4f}+/-{r.sobel_std:.4f}  "
              f"HFlog={r.hf_log_ratio_mean:+.4f}+/-{r.hf_log_ratio_std:.4f}")

    # Sanity check: report whether monotonicity holds (Sobel should decrease
    # with blur, increase with sharpen; HF-FFT log-ratio should follow the
    # same direction since it is *relative* to the untreated reference).
    def monotone(values, expect_increasing: bool) -> bool:
        for a, b in zip(values, values[1:]):
            if expect_increasing and not (b >= a - 1e-4):
                return False
            if not expect_increasing and not (b <= a + 1e-4):
                return False
        return True

    print("\n=== monotonicity checks ===")
    print(f"  Sobel decreases with blur:      "
          f"{monotone([r.sobel_mean for r in blur_rows], expect_increasing=False)}")
    print(f"  HFlog decreases with blur:      "
          f"{monotone([r.hf_log_ratio_mean for r in blur_rows], expect_increasing=False)}")
    print(f"  Sobel increases with sharpen:   "
          f"{monotone([r.sobel_mean for r in sharp_rows], expect_increasing=True)}")
    print(f"  HFlog increases with sharpen:   "
          f"{monotone([r.hf_log_ratio_mean for r in sharp_rows], expect_increasing=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
