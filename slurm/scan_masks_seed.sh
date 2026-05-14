#!/bin/bash
#SBATCH --job-name=scan_masks
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/scan_masks_%j.out
#SBATCH --error=logs/scan_masks_%j.err
#
# Scan masks/*.png for one seed's results and emit a per-(video, obj_id)
# CSV with the in-anchor-span empty-frame breakdown. CPU-only.
#
# Submit:  SEED=1 sbatch slurm/scan_masks_seed.sh
# Output:  results/_mask_scan_seed${SEED}.csv  (rsync to olab-1 when done)

set -euo pipefail
SEED=${SEED:-1}

REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs

source .venv/bin/activate
python tools/scan_masks.py \
    --results-root results \
    --prompts-dir  prompts \
    --seed "$SEED" \
    > "results/_mask_scan_seed${SEED}.csv"

echo "scan_masks_seed.sh done for SEED=$SEED at $(date -Iseconds)"
