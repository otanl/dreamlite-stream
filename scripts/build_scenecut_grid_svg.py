"""Build scenecut_grid.svg — 2 rows (N=8, N=1) x 8 frames (28..35), cut at 32."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = _ROOT / "figures_for_paper" / "scenecut"
OUT_SVG = _ROOT / "figures_for_paper" / "scenecut_grid.svg"

FRAMES = list(range(28, 36))  # 28..35
ROWS = [
    ("N = 8 (default)", "N8", "stale cond_emb from batch 0 (pre-cut)"),
    ("N = 1 (oracle)",  "N1", "always-refresh; equivalent to perfect cut detector"),
]

CELL = 192
PAD_X = 2
PAD_Y = 4
LBL_W = 200
HDR_H = 38       # column header + pre/cut/post marker band
TITLE_H = 36
CUT_AT = 32

PAGE_W = LBL_W + 8 * CELL + 7 * PAD_X + 12
PAGE_H = TITLE_H + HDR_H + 2 * CELL + PAD_Y + 36


def main():
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{PAGE_W}" height="{PAGE_H}" viewBox="0 0 {PAGE_W} {PAGE_H}">',
        '<style>',
        'text { font-family: "Charter", "Source Serif Pro", "Georgia", "Times New Roman", serif; fill: #111; }',
        '.title { font-size: 15px; font-weight: 700; }',
        '.subtitle { font-size: 11.5px; fill: #555; }',
        '.lbl { font-size: 14px; font-weight: 700; }',
        '.sub { font-size: 10.5px; fill: #555; font-style: italic; }',
        '.col { font-size: 11px; font-weight: 600; }',
        '.tag { font-size: 10.5px; font-weight: 700; fill: #6b5e44; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    # Title + subtitle
    lines.append(
        f'<text class="title" x="{PAGE_W//2}" y="20" text-anchor="middle">'
        f'Scene-cut at frame 32 on synthetic blackswan&#8594;goat clip</text>'
    )
    lines.append(
        f'<text class="subtitle" x="{PAGE_W//2}" y="36" text-anchor="middle">'
        f'B=8 batch boundary aligned with stale-cond window; '
        f'cut between frame 31 (pre, swan) and frame 32 (post, goat)</text>'
    )

    # Column headers (frame ids + pre/post tag band)
    band_y = TITLE_H + 14
    for i, f in enumerate(FRAMES):
        cx = LBL_W + i * (CELL + PAD_X) + CELL // 2
        is_post = f >= CUT_AT
        tag = "post-cut" if is_post else "pre-cut"
        # Frame number
        lines.append(
            f'<text class="col" x="{cx}" y="{band_y}" text-anchor="middle">'
            f'frame {f}</text>'
        )
        # Pre/post tag
        lines.append(
            f'<text class="sub" x="{cx}" y="{band_y+12}" text-anchor="middle" '
            f'fill="{"#a04a23" if is_post else "#3a5c91"}">{tag}</text>'
        )

    # Vertical cut indicator between frame 31 and 32
    cut_col_idx = FRAMES.index(CUT_AT)
    cut_x = LBL_W + cut_col_idx * (CELL + PAD_X) - PAD_X / 2
    cut_top = TITLE_H + HDR_H - 2
    cut_bot = TITLE_H + HDR_H + 2 * CELL + PAD_Y + 4
    lines.append(
        f'<line x1="{cut_x}" y1="{cut_top}" x2="{cut_x}" y2="{cut_bot}" '
        f'stroke="#c0392b" stroke-width="2.0" stroke-dasharray="6 3"/>'
    )
    lines.append(
        f'<text x="{cut_x}" y="{cut_top - 4}" text-anchor="middle" '
        f'class="tag" fill="#c0392b">CUT</text>'
    )

    # 2 rows
    for r, (lbl, stub, sub) in enumerate(ROWS):
        y = TITLE_H + HDR_H + r * (CELL + PAD_Y)
        ly = y + CELL // 2 - 4
        lines.append(
            f'<text class="lbl" x="{LBL_W - 14}" y="{ly}" text-anchor="end">{lbl}</text>'
        )
        lines.append(
            f'<text class="sub" x="{LBL_W - 14}" y="{ly + 16}" text-anchor="end">{sub}</text>'
        )
        for i, f in enumerate(FRAMES):
            x = LBL_W + i * (CELL + PAD_X)
            png = FRAMES_DIR / f"{stub}_f{f:02d}.png"
            if not png.exists():
                lines.append(
                    f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                    f'fill="#f0f0f0" stroke="#aaa"/>'
                )
                continue
            href = png.relative_to(OUT_SVG.parent).as_posix()
            lines.append(
                f'<image x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                f'xlink:href="{href}" preserveAspectRatio="xMidYMid slice"/>'
                f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                f'fill="none" stroke="#cccccc" stroke-width="0.8"/>'
            )

    # Bottom note
    note_y = TITLE_H + HDR_H + 2 * CELL + PAD_Y + 22
    lines.append(
        f'<text class="subtitle" x="{LBL_W}" y="{note_y}" text-anchor="start">'
        f'In the N=8 row, frames 32&#8211;39 reuse the pre-cut cond_emb (computed at batch 0, frames 0&#8211;7); in the N=1 row, batch 4 refreshes on the post-cut content. '
        f'&#949;<tspan baseline-shift="sub" font-size="8.5">w</tspan> cannot distinguish the two; LPIPS to the N=1 oracle does (Tables 19, 20).'
        f'</text>'
    )

    lines.append('</svg>')

    OUT_SVG.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_SVG}  ({PAGE_W}x{PAGE_H})")


if __name__ == "__main__":
    main()
