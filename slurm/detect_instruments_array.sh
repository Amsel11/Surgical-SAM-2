#!/bin/bash
#SBATCH --job-name=gd_detect
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gd_detect_%A_%a.out
#SBATCH --error=logs/gd_detect_%A_%a.err
#SBATCH --array=0-29%8
#
# Stage-2 detection: generic-query Grounding DINO over each clip, sampled every
# EVERY_SEC seconds, overlapping boxes NMS-consolidated. Decoupled from OCR/SAM
# so the anchor frequency can be ablated.
#
# Full run:        sbatch slurm/detect_instruments_array.sh
# Ablate density:  EVERY_SEC=5 sbatch slurm/detect_instruments_array.sh
# One clip:        CLIP=/path/to/clip.mp4 sbatch --array=0 slurm/detect_instruments_array.sh
# Finetuned GD:    GD_CKPT=/path/to/ft sbatch slurm/detect_instruments_array.sh

set -euo pipefail
REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs
source .sam3_venv/bin/activate
export BP_REPO="$REPO"
export HF_HOME="$REPO/.hf_cache"
mkdir -p "$HF_HOME"

EVERY_SEC="${EVERY_SEC:-2}"
CLIPS="${CLIPS_DIR:-/gpfs/data/oermannlab/users/schula12/whipple-transfer/clips}"
OUT="$REPO/results/detect_e${EVERY_SEC}"

# CLIP env overrides the array selection (for a single-clip validation).
if [ -n "${CLIP:-}" ]; then
    SELECTED="$CLIP"
else
    mapfile -t VIDS < <(find "$CLIPS" -maxdepth 1 -type f -iname '*.mp4' | sort)
    if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${#VIDS[@]}" ]; then
        echo "task ${SLURM_ARRAY_TASK_ID} out of range (${#VIDS[@]} clips)"; exit 0
    fi
    SELECTED="${VIDS[$SLURM_ARRAY_TASK_ID]}"
fi

CKPT_ARG=()
[ -n "${GD_CKPT:-}" ] && CKPT_ARG=(--gd-checkpoint "$GD_CKPT")

echo "=== detect $(basename "$SELECTED") on $(hostname), every ${EVERY_SEC}s ==="
nvidia-smi | head -12 || true
python tools/detect_instruments_video.py \
    --video "$SELECTED" --out-dir "$OUT" \
    --every-sec "$EVERY_SEC" "${CKPT_ARG[@]}"
echo "done $(basename "$SELECTED") at $(date -Iseconds)"
