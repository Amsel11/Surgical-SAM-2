#!/bin/bash
#SBATCH --job-name=auto_track
#SBATCH --partition=superpod
#SBATCH --nodelist=sp-0010
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-37%8
#SBATCH --output=logs/auto_track_%A_%a.out
#SBATCH --error=logs/auto_track_%A_%a.err
#
# Fully-automated ZERO-SHOT pipeline: OCR segments -> per-segment generic-query
# Grounding DINO -> persistent IoU-matched tracks -> SurgSAM-2 bidirectional
# propagation -> dedup. One video per array task, 8 concurrent on SP-0010's
# 8x H100 (SuperPOD; explicitly authorized for this run).
#
# Uses .sam3_venv (has sam2 + sam3 + transformers/GD in one env).
# Segments come from the completed PaddleOCR run (all 38 whip videos).
#
# Validate a couple:  sbatch --array=0-1 slurm/auto_track_sp0010.sh
# Full whip pilot:     sbatch slurm/auto_track_sp0010.sh
# Segmenter ablation:  add  stage2_inference.model=sam3  (or sam2_oob) to the python line
set -euo pipefail
REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"; mkdir -p logs
source .sam3_venv/bin/activate

export BP_REPO="$REPO"
export BP_FRAMES_ROOT=/gpfs/data/oermannlab/private_data/whip/frames_attempt2
export BP_RESULTS_ROOT="$REPO/results"
export SURGSAM_MANIFEST="$REPO/manifest.db"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEGROOT="$REPO/results/ocr_slots_timeline_paddle_full"
mapfile -t VIDEOS < <(ls "$SEGROOT"/*/segments.csv 2>/dev/null \
    | sed -E 's#.*/([^/]+)/segments\.csv$#\1#' | sort)
if [ "${#VIDEOS[@]}" -eq 0 ]; then echo "FATAL: no segments.csv under $SEGROOT"; exit 2; fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#VIDEOS[@]}" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID >= ${#VIDEOS[@]} videos; nothing to do."; exit 0
fi
VID="${VIDEOS[$SLURM_ARRAY_TASK_ID]}"

echo "=== task $SLURM_ARRAY_TASK_ID/${#VIDEOS[@]} -> $VID on $(hostname) @ $(date -Iseconds) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m pipeline run \
    +experiment=whip_auto_surgsam2 \
    "scope.videos=[$VID]" \
    scope.seed=1 \
    "stage1_prompting.segments_root=$SEGROOT"

echo "=== done $VID @ $(date -Iseconds) ==="
