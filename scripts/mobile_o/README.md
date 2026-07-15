# Mobile-O integration scripts (secondary substrate)

Canonical versioned copies. **Execution context is `F:\work\Mobile-O`**
(the cloned upstream repo + `.venv-mobileo`), not this directory — the
scripts import the `mobileo` package and resolve `checkpoints/`,
`assets/`, and `smoke_outputs/` relative to that root.

## Run environment

`F:\work\Mobile-O\.venv-mobileo` — Python 3.12, torch 2.3.0+cu121,
diffusers 0.35.2, transformers 4.51.3, timm 1.0.27 (upstream's
`timm==0.6.13` pin is incompatible with their own vendored MobileCLIP).
xformers / flash_attn / deepspeed are intentionally excluded
(SDPA fallback; see `notes/mobile_o_smoke_test_plan.md`).

```powershell
cd F:\work\Mobile-O
.venv-mobileo\Scripts\python.exe <script>.py
```

## Scripts

| Script | Purpose | Status marker |
|---|---|---|
| `run_smoke_test.py` | 3-mode smoke (T2I / understanding / editing) | `SMOKE_PASS` |
| `streaming_recipe.py` | R1 (CondRefreshCache) + R2 (SideStreamTE) functional checks + reduced-step denoise | `R1R2_FUNCTIONAL_PASS` |
| `test_idpruner_mobileo.py` | R3: IDPruner fixed-K budget on the projected visual tokens (K sweep 32/64/128) | `R3_MOBILEO_PASS` |
| `mobile_o_video.py` | First video-to-video: 3-mode comparison (naive / frozen / SDEdit) on DAVIS | `MOBILE_O_VIDEO_DONE` |

All four passed on RTX 3090 Ti, 2026-06-11. Findings and informal
numbers: `notes/mobile_o_smoke_test_plan.md`. License constraints
(cc-by-nc-4.0, research-only): `notes/mobile_o_license_check.md`.

When editing, treat `F:\work\Mobile-O\*.py` as the working copies and
re-sync here before committing.
