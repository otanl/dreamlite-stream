"""Build qualitative_grid.svg — 5 rows (input + 4 styles) x 3 clips."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = _ROOT / "figures_for_paper" / "qualitative_grid"
OUT_SVG = _ROOT / "figures_for_paper" / "qualitative_grid.svg"

CLIPS = [("blackswan", "blackswan"), ("dance-twirl", "dance-twirl"), ("parkour", "parkour")]
ROWS = [
    ("Input (DAVIS)", "input"),
    ("Oil-painting", "oil_champion"),
    ("Comic (held-out)", "comic_heldout"),
    ("Ukiyo-e (held-out)", "ukiyoe_heldout"),
    ("Van Gogh (held-out)", "vangogh_heldout"),
]

CELL = 256       # image cell side (pixels)
PAD = 4          # gap between cells
LBL_W = 168      # left label column width
HDR_H = 28       # column header height
TITLE_H = 32
PAGE_W = LBL_W + 3 * CELL + 2 * PAD + 8
PAGE_H = TITLE_H + HDR_H + 5 * CELL + 4 * PAD + 8

SUBTLE = "#555"
ACCENT = "#1d2f48"


def main():
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{PAGE_W}" height="{PAGE_H}" viewBox="0 0 {PAGE_W} {PAGE_H}">',
        '<style>',
        'text { font-family: "Charter", "Source Serif Pro", "Georgia", "Times New Roman", serif; fill: #111; }',
        '.lbl { font-size: 14px; }',
        '.sub { font-size: 10.5px; fill: #555; font-style: italic; }',
        '.hdr { font-size: 13px; font-weight: 700; }',
        '.title { font-size: 15px; font-weight: 700; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    # Title
    lines.append(
        f'<text class="title" x="{PAGE_W//2}" y="20" text-anchor="middle">'
        f'Held-out style generalisation of the v3 Temporal LLLite</text>'
    )
    # Column headers
    for i, (label, _stub) in enumerate(CLIPS):
        cx = LBL_W + i * (CELL + PAD) + CELL // 2
        cy = TITLE_H + 18
        lines.append(
            f'<text class="hdr" x="{cx}" y="{cy}" text-anchor="middle">{label}</text>'
        )
    # Rows
    for r, (lbl, stub) in enumerate(ROWS):
        y = TITLE_H + HDR_H + r * (CELL + PAD)
        # Row label
        ly = y + CELL // 2 - 4
        lines.append(
            f'<text class="lbl" x="{LBL_W - 12}" y="{ly}" text-anchor="end" '
            f'font-weight="{"700" if r == 0 else "500"}">{lbl}</text>'
        )
        # Trained vs. held-out tag below row label
        tag = "training prompt" if stub == "oil_champion" else (
              "source video" if stub == "input" else "unseen during training")
        lines.append(
            f'<text class="sub" x="{LBL_W - 12}" y="{ly + 16}" text-anchor="end">{tag}</text>'
        )
        # 3 image cells
        for c, (_label, slug) in enumerate(CLIPS):
            x = LBL_W + c * (CELL + PAD)
            png = FRAMES_DIR / f"{stub}_{slug}.png"
            if not png.exists():
                lines.append(
                    f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                    f'fill="#f0f0f0" stroke="#aaa"/>'
                    f'<text x="{x+CELL//2}" y="{y+CELL//2}" text-anchor="middle" '
                    f'class="sub">missing</text>'
                )
                continue
            # Use relative href so inkscape resolves on disk
            href = png.relative_to(OUT_SVG.parent).as_posix()
            lines.append(
                f'<image x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                f'xlink:href="{href}" preserveAspectRatio="xMidYMid slice"/>'
                f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                f'fill="none" stroke="#cccccc" stroke-width="0.8"/>'
            )
    lines.append('</svg>')

    OUT_SVG.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_SVG}  ({PAGE_W}x{PAGE_H})")


if __name__ == "__main__":
    main()
