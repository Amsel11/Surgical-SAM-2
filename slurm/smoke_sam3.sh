#!/bin/bash
#SBATCH --job-name=sam3_smoke
#SBATCH --partition=oermannlab,a100_short
#SBATCH --exclude=a100-8003
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sam3_smoke_%j.out
#SBATCH --error=logs/sam3_smoke_%j.err
#
# One-video SAM 3 inference to validate the new tracker end-to-end before
# kicking off a full array. Uses the shortest seed_2 video (DC_whip_11609423,
# 2590 frames) and writes outputs under results/sam3_smoke/.
#
# Submit:  sbatch slurm/smoke_sam3.sh

set -euo pipefail

REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs

source .sam3_venv/bin/activate

export BP_REPO="$REPO"
export BP_FRAMES_ROOT=/gpfs/data/oermannlab/private_data/whip/frames_attempt2
export BP_RESULTS_ROOT="$REPO/results"
export SURGSAM_MANIFEST="$REPO/local_manifest.db"

echo "=== env ==="
echo "python: $(which python)"
echo "node:   $(hostname)"
nvidia-smi | head -12 || true
echo

python -m pipeline run \
    +experiment=sam3_oob_whip \
    name=sam3_smoke \
    scope.seed=2 \
    'scope.videos=[DC_whip_11609423]'

echo
echo "smoke_sam3.sh done at $(date -Iseconds)"
echo "Output: results/sam3_smoke/DC_whip_11609423/"
