"""Build b3_o3_grid.svg — 3 rows (Input / B3 / O3) x 3 clips.

Each cell at 320 px; per-clip Sobel delta annotated under the column header.
"""
from __future__ import annotations

import base64
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = _ROOT / "figures_for_paper" / "b3_o3_grid"
OUT_SVG = _ROOT / "figures_for_paper" / "b3_o3_grid.svg"
OUT_PDF = OUT_SVG.with_suffix(".pdf")

CLIPS = [
    ("swing",   "Sobel 1.87 → 2.30 (+23.2%)"),
    ("parkour", "Sobel 2.14 → 2.64 (+23.1%)"),
    ("libby",   "Sobel 2.21 → 2.68 (+21.4%)"),
]
ROWS = [
    ("Input (DAVIS)", "input", "source frame"),
    ("B3 baseline",    "B3",    "vanilla MSE (= §V v3)"),
    ("O3 proposed",    "O3",    "MSE + LPIPS + spec"),
]

CELL = 320
PAD = 4
LBL_W = 184
HDR_H = 44   # taller header to fit Sobel delta line
TITLE_H = 36

PAGE_W = LBL_W + 3 * CELL + 2 * PAD + 8
PAGE_H = TITLE_H + HDR_H + 3 * CELL + 2 * PAD + 8


def _embed_png(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def main():
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{PAGE_W}" height="{PAGE_H}" viewBox="0 0 {PAGE_W} {PAGE_H}">',
        '<style>',
        'text { font-family: "Charter", "Source Serif Pro", "Georgia", "Times New Roman", serif; fill: #111; }',
        '.lbl { font-size: 15px; }',
        '.sub { font-size: 11px; fill: #555; font-style: italic; }',
        '.hdr { font-size: 14px; font-weight: 700; }',
        '.sobel { font-size: 11.5px; fill: #1d2f48; font-weight: 600; }',
        '.title { font-size: 16px; font-weight: 700; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    lines.append(
        f'<text class="title" x="{PAGE_W//2}" y="22" text-anchor="middle">'
        f'Smoothing collapse reversed: B3 (vanilla MSE) vs O3 (MSE + LPIPS + spectral)'
        f'</text>'
    )
    # Column headers (clip name + Sobel delta)
    for i, (slug, sobel) in enumerate(CLIPS):
        cx = LBL_W + i * (CELL + PAD) + CELL // 2
        cy = TITLE_H + 16
        lines.append(
            f'<text class="hdr" x="{cx}" y="{cy}" text-anchor="middle">{slug}</text>'
        )
        lines.append(
            f'<text class="sobel" x="{cx}" y="{cy + 16}" text-anchor="middle">{sobel}</text>'
        )
    # Rows
    for r, (lbl, stub, sub) in enumerate(ROWS):
        y = TITLE_H + HDR_H + r * (CELL + PAD)
        ly = y + CELL // 2 - 6
        lines.append(
            f'<text class="lbl" x="{LBL_W - 14}" y="{ly}" text-anchor="end" '
            f'font-weight="700">{lbl}</text>'
        )
        lines.append(
            f'<text class="sub" x="{LBL_W - 14}" y="{ly + 16}" text-anchor="end">{sub}</text>'
        )
        for c, (slug, _) in enumerate(CLIPS):
            x = LBL_W + c * (CELL + PAD)
            png = FRAMES_DIR / f"{stub}_{slug}.png"
            if not png.exists():
                lines.append(
                    f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                    f'fill="#f0f0f0" stroke="#aaa"/>'
                )
                lines.append(
                    f'<text x="{x+CELL//2}" y="{y+CELL//2}" text-anchor="middle">'
                    f'missing</text>'
                )
                continue
            href = _embed_png(png)
            lines.append(
                f'<image x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                f'xlink:href="{href}"/>'
            )
            lines.append(
                f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                f'fill="none" stroke="#222" stroke-width="0.7"/>'
            )
    lines.append('</svg>')
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUT_SVG.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {OUT_SVG}  ({OUT_SVG.stat().st_size // 1024} KiB)")

    # rsvg-convert -> PDF if available
    import shutil
    import subprocess
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        subprocess.run(
            [rsvg, "-f", "pdf", "-o", str(OUT_PDF), str(OUT_SVG)],
            check=False,
        )
        if OUT_PDF.exists():
            print(f"[saved] {OUT_PDF}  ({OUT_PDF.stat().st_size // 1024} KiB)")
    else:
        # Try inkscape as fallback
        ink = shutil.which("inkscape")
        if ink:
            subprocess.run(
                [ink, "--export-type=pdf", f"--export-filename={OUT_PDF}", str(OUT_SVG)],
                check=False,
            )
            if OUT_PDF.exists():
                print(f"[saved] {OUT_PDF}  ({OUT_PDF.stat().st_size // 1024} KiB)")
        else:
            print("[skip] no rsvg-convert or inkscape - open SVG in browser for PDF export")


if __name__ == "__main__":
    main()
