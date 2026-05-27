#!/bin/bash
#SBATCH --job-name=surgsam2_seed1
#SBATCH --partition=oermannlab,a100_short
#SBATCH --exclude=a100-8003
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-37%8
#SBATCH --output=logs/surgsam2_seed1_%A_%a.out
#SBATCH --error=logs/surgsam2_seed1_%A_%a.err
#
# SurgSAM-2 inference at scope.seed=1 against the relabeled whip prompts.
# Existing SurgSAM-2/SAM2 runs were at scope.seed=2 (per array_sam2_oob_sp.sh);
# Phase 0b's --edit pass writes labels to seed=1 manual_box, so we need a
# fresh inference pass at seed=1 for extract_ft_labels.py to join correctly.
#
# Outputs land at results/surgsam2_oob_whip_v1/<video>/seed_1/masks/*.png —
# the palette-PNG format extract_ft_labels expects.
#
# Prereq:  bp:manifest.db must have the relabeled prompt_objects rows
#          (rsync local_manifest.db → bp before submitting).
#
# Submit:  sbatch slurm/array_surgsam2_oob_whip_seed1.sh
#
# The array size 0-37 is sized to 38 videos in the whip cohort; %8 = 8 tasks
# concurrent at once. If videos lack a seed=1 prompts JSON, the orchestrator
# raises RuntimeError and the task fails fast — those are the relabel gaps
# (no seed=1 prompt_set yet); skip via slurm/array_sam2_oob_sp.sh-style retry
# pattern once they're labelled.

set -euo pipefail

REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs

source .venv/bin/activate

export BP_REPO="$REPO"
export BP_FRAMES_ROOT=/gpfs/data/oermannlab/private_data/whip/frames_attempt2
export BP_RESULTS_ROOT="$REPO/results"
export SURGSAM_MANIFEST="$REPO/manifest.db"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Sorted list of videos that have a seed=1 manual_box prompts JSON. We index
# into THIS list (not raw videos table) so the array maps cleanly onto videos
# whose Phase 0b labels are written. Videos still pending relabel are absent
# and won't run in this array.
mapfile -t VIDEOS < <(ls prompts/*_seed1_manual_box.json 2>/dev/null | sort | sed -E 's|prompts/||; s|_seed1_manual_box\.json||')

if [ "${#VIDEOS[@]}" -eq 0 ]; then
    echo "FATAL: no prompts/*_seed1_manual_box.json found. Did you sync the manifest + JSONs to bp?"
    exit 2
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#VIDEOS[@]}" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID >= ${#VIDEOS[@]} videos. Resize --array."
    exit 0
fi

VIDEO_ID="${VIDEOS[$SLURM_ARRAY_TASK_ID]}"

echo "=== task $SLURM_ARRAY_TASK_ID / ${#VIDEOS[@]} -> $VIDEO_ID on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

python -m pipeline run \
    +experiment=surgsam2_oob_whip \
    scope.seed=1 \
    "scope.videos=[$VIDEO_ID]"

echo
echo "task $SLURM_ARRAY_TASK_ID ($VIDEO_ID) done at $(date -Iseconds)"
