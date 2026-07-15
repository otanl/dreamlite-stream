#!/usr/bin/env bash
# Post-training eval + comparison for the O3-pix retrain (#157).
#
# Run after runs/lcm_lora_O3pix/lcm_lora_step001644.pt lands.
# Generates O3pix inference outputs on the 9 DAVIS sequences then
# re-runs measure_smoothing_collapse over all 5 configs.
#
# Expected total runtime ~15 minutes on RTX 3090 Ti.
set -e

cd "$(dirname "$0")/.."

CKPT="runs/lcm_lora_O3pix/lcm_lora_step001644.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: $CKPT not found. Wait for training to finish."
    exit 1
fi

echo "=== Step 1: O3pix inference on DAVIS-2017 (9 sequences) ==="
python scripts/eval_lcm_lora.py \
    --lcm_lora_weights "$CKPT" \
    --out_dir out/compare_b3_vs_o3/O3pix \
    --seed 42 \
    --size 512 \
    --steps 1 \
    --cond_refresh_every 8 \
    --max_frames 48

echo
echo "=== Step 2: Smoothing-collapse measurement on 5 configs ==="
python scripts/measure_smoothing_collapse.py \
    --root out/compare_b3_vs_o3 \
    --configs B3 O1 O2 O3 O3pix \
    --davis-root assets/davis/DAVIS/JPEGImages/480p \
    --max-frames 48 \
    --resolution 512 \
    --out-dir out/smoothing_rederivation_v2

echo
echo "=== Step 3: Summary at out/smoothing_rederivation_v2/summary.md ==="
cat out/smoothing_rederivation_v2/summary.md

echo
echo "=== Hypothesis verdict ==="
python -c "
import csv
rows = list(csv.DictReader(open('out/smoothing_rederivation_v2/per_config.csv')))
by_cfg = {r['config']: r for r in rows}
o1 = float(by_cfg['O1']['hf_log_ratio_mean'])
o3 = float(by_cfg['O3']['hf_log_ratio_mean'])
o3pix = float(by_cfg.get('O3pix', {}).get('hf_log_ratio_mean', 0))
print(f'O1 HFlog (LPIPS only):           {o1:+.4f}  (W1 best)')
print(f'O3 HFlog (LPIPS + spec-latent):  {o3:+.4f}  (W1 regression)')
print(f'O3pix HFlog (LPIPS + spec-pixel):{o3pix:+.4f}  (W5 hypothesis test)')
if o3pix > o1:
    print()
    print('  VERDICT: O3pix > O1.  Pixel-space spectral loss recovers')
    print('           more source HF than LPIPS alone. Hypothesis CONFIRMED.')
elif o3pix > o3 and o3pix > o3 + 0.02:
    print()
    print('  VERDICT: O3pix > O3 but < O1.  Pixel-space helps over latent')
    print('           but does not surpass LPIPS-only. Hypothesis PARTIAL.')
else:
    print()
    print('  VERDICT: O3pix <= O3.  Pixel-space variant does not help.')
    print('           Hypothesis REJECTED -- spectral term is not the right')
    print('           additional component on top of LPIPS.')
"
