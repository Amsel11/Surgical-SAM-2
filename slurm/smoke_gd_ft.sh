#!/bin/bash
#SBATCH --job-name=gdino_ft_smoke
#SBATCH --partition=oermannlab,a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/gdino_ft_smoke_%j.out
#SBATCH --error=logs/gdino_ft_smoke_%j.err
#
# Smoke test for the HF GroundingDINO fine-tune plumbing (Phase 7 verification).
# Overfits a handful of train images for 50 steps and saves nothing — the goal
# is to confirm the data -> processor -> model -> loss path runs end-to-end and
# the training loss drops, BEFORE committing to the 4-hour slurm/ft_gd_train.sh.
#
# Watch logs/gdino_ft_smoke_*.out: 'loss' on the logged lines should fall over
# the 50 steps. A crash here means the dataset/collate/label wiring is off.
#
# Prereqs: same as ft_gd_train.sh (setup_gdino_ft_bp.sh + a built data/ft_hf_v1).
#
# Submit:  sbatch slurm/smoke_gd_ft.sh

set -euo pipefail

REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs

source .gdino_venv/bin/activate

DATA_DIR="$REPO/data/ft_hf_v1"
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "MISSING $DATA_DIR/train.jsonl — build it with tools.ft_dataset_convert first."
    exit 1
fi

python -m tools.ft_gd_train \
    --data-dir "$DATA_DIR" \
    --output-dir "/tmp/gdino_ft_smoke_$$" \
    --max-steps 50 \
    --limit-train 4 \
    --batch-size 2 \
    --lr 1e-4 \
    --logging-steps 5

echo "Smoke done. If loss fell and no crash, slurm/ft_gd_train.sh is good to go."
