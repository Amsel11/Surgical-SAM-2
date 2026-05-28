#!/bin/bash
#SBATCH --job-name=whip_auto_smoke
#SBATCH --partition=superpod
#SBATCH --nodelist=sp-0011
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/whip_auto_smoke_%j.out
#SBATCH --error=logs/whip_auto_smoke_%j.err
#
# Single-video smoke test of the auto pipeline — validates the full
# OCR -> auto-GD -> SurgSAM-2 -> dedup path on one video before the array.
#
# Submit:        sbatch slurm/whip_auto_smoke.sh                 # DC_whip_11609423
# Other video:   VID=ER_whip_xxxx sbatch slurm/whip_auto_smoke.sh

set -euo pipefail
REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs
source .sam3_venv/bin/activate
export BP_REPO="$REPO"
export SURGSAM_MANIFEST="$REPO/manifest.db"
export HF_HOME="$REPO/.hf_cache"
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPERIMENT="${EXPERIMENT:-whip_auto_surgsam2}"
VID="${VID:-DC_whip_11609423}"

echo "=== smoke: ${VID} via +experiment=${EXPERIMENT} on $(hostname) ==="
nvidia-smi | head -12 || true
echo

python -m pipeline run "+experiment=${EXPERIMENT}" "scope.videos=[${VID}]"

echo
echo "smoke (${VID}) done at $(date -Iseconds)"
