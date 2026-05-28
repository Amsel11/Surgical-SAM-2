#!/bin/bash
#SBATCH --job-name=whip_auto_smoke
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/whip_auto_smoke_%j.out
#SBATCH --error=logs/whip_auto_smoke_%j.err
#
# a100 variant of the single-video smoke test — for when the SuperPOD nodes
# (sp-0011/sp-0012) are GPU-saturated by other users. No node pin; the
# scheduler picks any free a100. Same pipeline as whip_auto_smoke.sh.
#
# Submit:        sbatch slurm/whip_auto_smoke_a100.sh                # DC_whip_11609423
# Other video:   VID=ER_whip_xxxx sbatch slurm/whip_auto_smoke_a100.sh

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

echo "=== smoke(a100): ${VID} via +experiment=${EXPERIMENT} on $(hostname) ==="
nvidia-smi | head -12 || true
echo

python -m pipeline run "+experiment=${EXPERIMENT}" "scope.videos=[${VID}]"

echo
echo "smoke (${VID}) done at $(date -Iseconds)"
