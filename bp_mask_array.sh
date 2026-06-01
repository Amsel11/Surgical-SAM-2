#!/bin/bash
# SLURM array: one PJA video per task, full-length OCR-anchored masking on an A100.
#SBATCH --job-name=ssam2mask
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/data/oermannlab/users/schula12/ssam2-masking/logs/mask_%A_%a.out
set -u
module load ffmpeg/8.0 2>/dev/null || module load ffmpeg/7.1.1 2>/dev/null

BASE=/gpfs/data/oermannlab/users/schula12/ssam2-masking
MAIN=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
CLIPS=/gpfs/data/oermannlab/users/schula12/whipple-transfer/clips
PY=$MAIN/.venv/bin/python
export HF_HOME=$BASE/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$BASE" || exit 1

VID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$BASE/video_list.txt")
echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> $VID  host=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES"
[ -z "$VID" ] && { echo "no video for this index"; exit 0; }
[ -f "$BASE/batch_out/$VID/DONE" ] && { echo "already DONE, skipping"; exit 0; }

$PY -m tools.auto_mask \
  --video "$CLIPS/$VID.mp4" --video-id "$VID" \
  --segments-csv "$BASE/all_segments_v2_burst.csv" \
  --start-sec 0 --end-sec 999999 \
  --out-dir "$BASE/batch_out/$VID" \
  --fps 6 --max-chunk-sec 150 --min-score 0.15 \
  --device cuda:0 \
  --checkpoint "$MAIN/checkpoints/sam2.1_hiera_s_endo18.pth" \
  && touch "$BASE/batch_out/$VID/DONE" && echo "[$(date)] DONE $VID" \
  || echo "[$(date)] FAILED $VID"
