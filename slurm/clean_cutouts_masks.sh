#!/bin/bash
#SBATCH --job-name=clean_masks
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-33%34
#SBATCH --output=logs/clean_masks_%A_%a.out
#SBATCH --error=logs/clean_masks_%A_%a.err
#
# Strip-clip + thin SurgSAM-2 masks and re-render clean cutouts, one task per
# video (robust to ssh dropout, unlike a long login-node loop). For each video:
#   1) clean_subsample_masks: zero bottom da Vinci strip + keep every 15th mask
#      -> seed_1/masks_clean/   (dense seed_1/masks/ left untouched)
#   2) render_object_cutouts --strip-frac 0.17: re-render clean QC cutouts
#
# Submit:  sbatch slurm/clean_cutouts_masks.sh
set -euo pipefail
REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs
source .venv/bin/activate
export SURGSAM_MANIFEST="$REPO/manifest.db"

RUN=surgsam2_oob_whip_v1
# Index into videos that actually have a dense masks/ dir for this run.
mapfile -t VIDEOS < <(ls -d results/$RUN/*/seed_1/masks 2>/dev/null \
    | sed -E 's|results/'"$RUN"'/||; s|/seed_1/masks||' | sort)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#VIDEOS[@]}" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID >= ${#VIDEOS[@]} videos; nothing to do."
    exit 0
fi
VID="${VIDEOS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task $SLURM_ARRAY_TASK_ID -> $VID on $(hostname) ==="

python -m tools.clean_subsample_masks --run "$RUN" --video "$VID" --seed 1 \
    --keep-stride 15 --strip-frac 0.17

python -m tools.render_object_cutouts --run "$RUN" --video "$VID" --seed 1 \
    --strip-frac 0.17

echo "task $SLURM_ARRAY_TASK_ID ($VID) done at $(date -Iseconds)"
