#!/bin/bash
# Batch auto-mask every raw_clips video, full length. Resumable + caffeinated.
# Knobs via env: FPS (default 6), CHUNK seconds (150), RECOVER (1=on,0=off).
set -u
REPO="/Users/schula12/OLAB/ML Projects/Surgical-SAM-2-masking"
CLIPS="/Users/schula12/whipple/raw_clips"
CSV="/Users/schula12/Desktop/all_segments_v2_burst.csv"
OUT="/Users/schula12/whipple/masking_test/batch_masks"
FPS="${FPS:-6}"; CHUNK="${CHUNK:-150}"; RECOVER="${RECOVER:-1}"
mkdir -p "$OUT"; cd "$REPO" || exit 1
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
recflag=""; [ "$RECOVER" = "0" ] && recflag="--no-recover"
echo "BATCH START $(date)  FPS=$FPS CHUNK=$CHUNK RECOVER=$RECOVER"
for vid in "$CLIPS"/*.mp4; do
  id="$(basename "$vid" .mp4)"
  if [ -f "$OUT/$id/DONE" ]; then echo "[skip] $id (already DONE)"; continue; fi
  echo "[start] $id  $(date)"
  if python -m tools.auto_mask --video "$vid" --video-id "$id" \
       --segments-csv "$CSV" --start-sec 0 --end-sec 999999 \
       --out-dir "$OUT/$id" --fps "$FPS" --max-chunk-sec "$CHUNK" \
       --min-score 0.15 $recflag; then
    touch "$OUT/$id/DONE"; echo "[done] $id  $(date)"
  else
    echo "[FAIL] $id (no segments in CSV, or error)  $(date)"
  fi
done
echo "BATCH COMPLETE $(date)"
