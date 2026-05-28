#!/bin/bash
#SBATCH --job-name=whip_auto
#SBATCH --partition=superpod
#SBATCH --nodelist=sp-0011
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/whip_auto_%A_%a.out
#SBATCH --error=logs/whip_auto_%A_%a.err
#SBATCH --array=0-37%8
#
# Canonical fully-automated pipeline (configs/experiment/whip_auto_surgsam2.yaml).
# One array task = one video, end-to-end via `python -m pipeline run`.
#
# Submit:           sbatch slurm/whip_auto_array.sh
# Re-run a subset:  sbatch --array=3,7,12 slurm/whip_auto_array.sh
# Ablate segmenter: EXPERIMENT=whip_auto_sam3 sbatch slurm/whip_auto_array.sh

set -euo pipefail
REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs
source .sam3_venv/bin/activate
export BP_REPO="$REPO"
# bp's populated manifest is manifest.db (local_manifest.db is olab-1 working
# state). run.py reads frames_dir from config.infrastructure.manifest_db, which
# interpolates ${oc.env:SURGSAM_MANIFEST,./local_manifest.db} — point it at the
# real one so the videos table lookups resolve.
export SURGSAM_MANIFEST="$REPO/manifest.db"
export HF_HOME="$REPO/.hf_cache"
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPERIMENT="${EXPERIMENT:-whip_auto_surgsam2}"
SEGMENTS_ROOT="$REPO/results/ocr_slots_timeline"

# Video list pulled from the OCR segments dir at submit time. Exclude the
# corrupt-frame video (NP_whip_16798242 has a 0-byte frame that crashes the
# SAM2 loader — see memory).
mapfile -t VIDS < <(ls "$SEGMENTS_ROOT" | grep -v '^NP_whip_16798242$' | sort)

if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${#VIDS[@]}" ]; then
    echo "task ${SLURM_ARRAY_TASK_ID} out of range (only ${#VIDS[@]} videos)"
    exit 0
fi
VID="${VIDS[$SLURM_ARRAY_TASK_ID]}"

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${VID} via +experiment=${EXPERIMENT} on $(hostname) ==="
nvidia-smi | head -12 || true
echo

python -m pipeline run "+experiment=${EXPERIMENT}" "scope.videos=[${VID}]"

echo
echo "task ${SLURM_ARRAY_TASK_ID} (${VID}) done at $(date -Iseconds)"
