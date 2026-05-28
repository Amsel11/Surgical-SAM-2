#!/bin/bash
#SBATCH --job-name=gdino_ft
#SBATCH --partition=oermannlab,a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gdino_ft_%j.out
#SBATCH --error=logs/gdino_ft_%j.err
#
# Fine-tune HF grounding-dino-tiny on the cardiac-whip detection dataset.
#
# Prereqs:
#   1. sbatch slurm/setup_gdino_ft_bp.sh        (adds FT extras to .gdino_venv,
#                                                pre-caches grounding-dino-tiny)
#   2. (in .sam3_venv) the Tier-B dataset exists:
#        python -m tools.extract_ft_labels --run surgsam2_oob_whip_v1 --seed 1 \
#            --max-per-video 200
#        python -m tools.make_split
#        python -m tools.ft_dataset_convert \
#            --in data/ft_labels_v1.jsonl --split configs/splits/whip_ft_v1.yaml \
#            --vocab configs/cardiac_whip_vocab.json --out data/ft_hf_v1
#
# Output: checkpoints/gdino_whip_ft_v1/  (a standard HF model dir — point
#         configs/experiment/ft_gd_whip_v1_inference.yaml's
#         stage1_prompting.checkpoint at it).
#
# Submit:  sbatch slurm/ft_gd_train.sh
#          sbatch slurm/ft_gd_train.sh --lora            # LoRA instead of full FT
#          sbatch slurm/ft_gd_train.sh --epochs 25 --lr 3e-5
#   (extra args after the script name pass straight through to tools.ft_gd_train)

set -euo pipefail

REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs checkpoints

source .gdino_venv/bin/activate

DATA_DIR="$REPO/data/ft_hf_v1"
if [ ! -f "$DATA_DIR/train.jsonl" ] || [ ! -f "$DATA_DIR/categories.json" ]; then
    echo "MISSING dataset in $DATA_DIR — run tools.ft_dataset_convert first (see header)."
    exit 1
fi

python -m tools.ft_gd_train \
    --data-dir "$DATA_DIR" \
    --output-dir "$REPO/checkpoints/gdino_whip_ft_v1" \
    --epochs 15 \
    --lr 5e-5 \
    --batch-size 2 \
    "$@"

echo
echo "Done. Fine-tuned model in checkpoints/gdino_whip_ft_v1/"
echo "Run the FT ablation arm:"
echo "  source .sam3_venv/bin/activate   # SAM3 stage-2 lives here"
echo "  python -m pipeline run +experiment=ft_gd_whip_v1_inference"
