#!/bin/bash
#SBATCH --job-name=render_overlay
#SBATCH --partition=cpu_medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/render_overlay_%j.out
#SBATCH --error=logs/render_overlay_%j.err
#
# CPU-only overlay rendering for one seed batch. Composites overlay frames
# from existing masks/ + source frames, encodes overlay.mp4 + preview_small.mp4.
#
# Submit with:
#   SEED=2 sbatch slurm/render_overlays_seed.sh
#
# Defaults to SEED=2. No GPU needed.

set -euo pipefail
SEED=${SEED:-2}

REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs

source .venv/bin/activate

python tools/render_overlay_from_masks.py \
    --batch results/ \
    --seed "$SEED" \
    --frames-root /gpfs/data/oermannlab/private_data/whip/frames_attempt2

echo "render_overlays_seed.sh done for SEED=$SEED at $(date -Iseconds)"
