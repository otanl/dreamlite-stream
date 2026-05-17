#!/bin/bash
# Phase 2 + Phase 3 on vast.ai 4090 (Ryzen 5900X)
set -e
cd /workspace
source venv/bin/activate

# -- Phase 2: cond-refresh sweep on down_blocks champion --
echo "=========================================="
echo "Phase 2: cond-refresh sweep on down_blocks"
echo "=========================================="
cd /workspace/dreamlite-stream
python -u scripts/cond_refresh_sweep_downblocks.py \
  --refresh_intervals 1 4 8 16 \
  --max_frames 64 \
  --out_dir /workspace/dreamlite-stream/out/cond_refresh_downblocks_sweep \
  2>&1 | tee /root/cond_refresh_sweep.log

# -- Phase 3: cross-dataset eval on Google commondatastorage sample mp4s --
echo ""
echo "=========================================="
echo "Phase 3: cross-dataset eval (Google sample mp4s)"
echo "=========================================="
mkdir -p /workspace/dreamlite-stream/assets/crossds_raw

# Google's commondatastorage gtv-videos-bucket has CC-licensed sample mp4s
# used by the Android/Google SDK. These have stable direct URLs.
# Mix of natural footage (cars, driving) and animation (Sintel, BBB excluded).
BASE="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample"
declare -A CLIPS
CLIPS["bigger_blazes"]="ForBiggerBlazes.mp4"
CLIPS["bigger_escapes"]="ForBiggerEscapes.mp4"
CLIPS["bigger_fun"]="ForBiggerFun.mp4"
CLIPS["bigger_joyrides"]="ForBiggerJoyrides.mp4"
CLIPS["bigger_meltdowns"]="ForBiggerMeltdowns.mp4"
CLIPS["subaru_offroad"]="SubaruOutbackOnStreetAndDirt.mp4"
CLIPS["vw_gti_review"]="VolkswagenGTIReview.mp4"
CLIPS["bullrun"]="WeAreGoingOnBullrun.mp4"
CLIPS["car_for_grand"]="WhatCarCanYouGetForAGrand.mp4"

cd /workspace/dreamlite-stream/assets/crossds_raw
for key in "${!CLIPS[@]}"; do
  src="${CLIPS[$key]}"
  if [ -f "${key}.mp4" ]; then continue; fi
  echo "  downloading ${key} <- $src"
  curl -L -o "${key}.mp4" "${BASE}/${src}" --silent --show-error || \
    echo "  WARN: failed ${key}"
done

echo "[downloaded clips]:"
ls -lh /workspace/dreamlite-stream/assets/crossds_raw/

# Preprocess: center-crop + scale to 512x512, trim to 2.7s @ 24fps = 64 frames
mkdir -p /workspace/dreamlite-stream/assets/crossds_512
for mp4 in /workspace/dreamlite-stream/assets/crossds_raw/*.mp4; do
  name=$(basename "$mp4")
  out="/workspace/dreamlite-stream/assets/crossds_512/$name"
  if [ -f "$out" ]; then continue; fi
  echo "  preprocessing $name"
  ffmpeg -i "$mp4" \
         -vf "crop='min(iw,ih)':'min(iw,ih)',scale=512:512" \
         -t 3 -r 24 -an -y "$out" 2>&1 | tail -2
done

echo "[preprocessed (cross-dataset clips)]:"
ls -lh /workspace/dreamlite-stream/assets/crossds_512/

# Run champion_eval on these clips
SEQS=""
for mp4 in /workspace/dreamlite-stream/assets/crossds_512/*.mp4; do
  name=$(basename "$mp4" .mp4)
  SEQS="$SEQS $name"
done
echo "[eval] sequences:$SEQS"

cd /workspace/dreamlite-stream
python -u scripts/champion_eval.py \
  --mp4_dir /workspace/dreamlite-stream/assets/crossds_512 \
  --sequences $SEQS \
  --max_frames 64 \
  --out_dir /workspace/dreamlite-stream/out/champion_crossds \
  2>&1 | tee /root/crossds_eval.log

echo ""
echo "=========================================="
echo "DONE — Phase 2 + Phase 3"
echo "=========================================="
echo "Results:"
echo "  Phase 2: /workspace/dreamlite-stream/out/cond_refresh_downblocks_sweep/results.jsonl"
echo "  Phase 3: /workspace/dreamlite-stream/out/champion_crossds/results.jsonl"
echo "  Phase 3 mp4s: /workspace/dreamlite-stream/out/champion_crossds/champion/*.mp4"
