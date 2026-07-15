"""Build b3_o3_grid.png — PIL-based fallback when SVG -> PDF tooling is unavailable.

Same layout as the SVG variant: 3 rows (Input/B3/O3) x 3 cols (swing/parkour/libby),
single frame each, with column headers, row labels, title, and per-cell Sobel delta.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = _ROOT / "figures_for_paper" / "b3_o3_grid"
OUT_PNG = _ROOT / "figures_for_paper" / "b3_o3_grid.png"

CLIPS = [
    ("swing",   "Sobel B3 2.04  O1 2.70 (+32%)  O2 2.09  O3 2.40"),
    ("parkour", "Sobel B3 2.07  O1 2.80 (+35%)  O2 2.10  O3 2.53"),
    ("libby",   "Sobel B3 2.14  O1 2.80 (+31%)  O2 2.14  O3 2.49"),
]
ROWS = [
    ("Input (DAVIS)",          "input",    "source frame"),
    ("Champion",               "champion", "DreamLite + LLLite, no LoRA"),
    ("B3 + LLLite",             "B3",       "MSE LCM-LoRA on top"),
    ("O1 + LLLite (best config)",  "O1",       "MSE + LPIPS only"),
    ("O2 + LLLite",             "O2",       "MSE + spectral only"),
    ("O3 + LLLite",             "O3",       "MSE + LPIPS + spec"),
]

CELL = 320
PAD = 4
LBL_W = 200
HDR_H = 48
TITLE_H = 38

PAGE_W = LBL_W + 3 * CELL + 2 * PAD + 12
PAGE_H = TITLE_H + HDR_H + 6 * CELL + 5 * PAD + 12


def _font(size: int, *, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    # Fallback chain — Windows default fonts that are always present.
    candidates = []
    if bold and italic:
        candidates += ["georgiabi.ttf", "timesbi.ttf"]
    if bold:
        candidates += ["georgiab.ttf", "timesbd.ttf", "arialbd.ttf"]
    if italic:
        candidates += ["georgiai.ttf", "timesi.ttf", "ariali.ttf"]
    candidates += ["georgia.ttf", "times.ttf", "arial.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(img)

    f_title  = _font(17, bold=True)
    f_hdr    = _font(15, bold=True)
    f_sobel  = _font(12, bold=True)
    f_lbl    = _font(16, bold=True)
    f_sub    = _font(12, italic=True)

    # Title
    title = "4-config ablation with LLLite oil-paint (DAVIS, K=1): LPIPS reverses smoothing, spec inactive"
    bbox = d.textbbox((0, 0), title, font=f_title)
    tw = bbox[2] - bbox[0]
    d.text(((PAGE_W - tw) // 2, 12), title, fill="black", font=f_title)

    # Column headers
    for i, (slug, sobel) in enumerate(CLIPS):
        cx = LBL_W + i * (CELL + PAD) + CELL // 2
        bx = d.textbbox((0, 0), slug, font=f_hdr)
        d.text((cx - (bx[2] - bx[0]) // 2, TITLE_H + 4), slug, fill="black", font=f_hdr)
        bs = d.textbbox((0, 0), sobel, font=f_sobel)
        d.text((cx - (bs[2] - bs[0]) // 2, TITLE_H + 26), sobel, fill="#1d2f48", font=f_sobel)

    # Rows
    for r, (lbl, stub, sub) in enumerate(ROWS):
        y = TITLE_H + HDR_H + r * (CELL + PAD)
        lb = d.textbbox((0, 0), lbl, font=f_lbl)
        d.text(
            (LBL_W - 14 - (lb[2] - lb[0]), y + CELL // 2 - 14),
            lbl, fill="black", font=f_lbl,
        )
        sb = d.textbbox((0, 0), sub, font=f_sub)
        d.text(
            (LBL_W - 14 - (sb[2] - sb[0]), y + CELL // 2 + 6),
            sub, fill="#555", font=f_sub,
        )
        for c, (slug, _) in enumerate(CLIPS):
            x = LBL_W + c * (CELL + PAD)
            png = FRAMES_DIR / f"{stub}_{slug}.png"
            if not png.exists():
                d.rectangle([x, y, x + CELL, y + CELL], outline="gray", fill="#f0f0f0")
                d.text((x + CELL // 2 - 30, y + CELL // 2 - 8), "missing", fill="red")
                continue
            cell = Image.open(png).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
            img.paste(cell, (x, y))
            d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline="#222", width=1)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PNG, optimize=True)
    print(f"[saved] {OUT_PNG}  ({OUT_PNG.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
