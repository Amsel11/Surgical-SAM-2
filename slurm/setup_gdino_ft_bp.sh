#!/bin/bash
#SBATCH --job-name=gdino_ft_setup
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gdino_ft_setup_%j.out
#SBATCH --error=logs/gdino_ft_setup_%j.err
#
# One-shot setup for Grounding DINO **fine-tuning**, HF-native path.
#
# We fine-tune via HuggingFace transformers (GroundingDinoForObjectDetection
# computes the bipartite-matching loss when given `labels`), NOT via
# Open-GroundingDino. Reason: the Stage-1 prompter (pipeline/prompts/dino.py)
# loads GD through transformers, so an HF-trained checkpoint drops straight into
# the prompter's `checkpoint` override with zero conversion. (An OGD-trained
# state-dict would need fragile key-renaming to load in HF.)
#
# This env REUSES .gdino_venv (the inference env from setup_gdino_bp.sh) and just
# adds the training extras. Sharing the env guarantees the transformers version
# used for training == the one the prompter loads with, so the saved model is
# always loadable.
#
# Submit:  sbatch slurm/setup_gdino_ft_bp.sh
#
# Then the end-to-end FT path (all in .gdino_venv unless noted):
#   # 1. (in .sam3_venv) build the dataset from propagated masks:
#   python -m tools.ft_dataset_convert \
#       --in data/ft_labels_v1.jsonl --split configs/splits/whip_ft_v1.yaml \
#       --vocab configs/cardiac_whip_vocab.json --out data/ft_hf_v1
#   # 2. fine-tune:
#   sbatch slurm/ft_gd_train.sh
#   # 3. point configs/experiment/ft_gd_whip_v1_inference.yaml's
#   #    stage1_prompting.checkpoint at checkpoints/gdino_whip_ft_v1
#
# NOTE: the older Open-GroundingDino clone/ODVG path this script used to set up
# is retired. If an Open-GroundingDino/ or .gdino_ft_venv/ dir is lying around
# from a previous run, it's unused now and safe to ignore (ask before deleting).

set -euo pipefail

REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs

# ── 1. Ensure the inference venv exists (build base deps if first run) ──
if [ ! -d .gdino_venv ]; then
    echo ".gdino_venv not found — building it (run slurm/setup_gdino_bp.sh first"
    echo "for the canonical build; doing a minimal build here)."
    PY="$(command -v python3.11 || command -v python3)"
    "$PY" -m venv .gdino_venv
    source .gdino_venv/bin/activate
    python -m pip install --upgrade pip
    pip install "torch>=2.4" "transformers>=4.51" "pillow>=10" "numpy<3" \
        "pydantic>=2.0" "omegaconf>=2.3" python-dotenv
else
    echo "Reusing existing .gdino_venv"
    source .gdino_venv/bin/activate
    python -m pip install --upgrade pip
fi

# ── 2. Add FT-only extras ─────────────────────────────────────────────
# accelerate    : transformers.Trainer backend
# albumentations: image+bbox augmentation (hflip / jitter)
# peft          : optional LoRA (--lora); harmless if unused
pip install "accelerate>=0.30" "albumentations>=1.4" "peft>=0.11"

# ── 3. Pre-cache the base model so GPU compute nodes (often no outbound
#       internet) can load it from the HF cache during training ──────────
python - <<'PY'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
mid = "IDEA-Research/grounding-dino-tiny"
print(f"Pre-caching {mid} ...")
AutoProcessor.from_pretrained(mid)
AutoModelForZeroShotObjectDetection.from_pretrained(mid)
print("cached OK")
PY

# ── 4. Smoke-test the training import chain ───────────────────────────
python - <<'PY'
import importlib
for m in ("torch", "transformers", "accelerate", "albumentations", "peft"):
    importlib.import_module(m)
    print(f"  import {m}: OK")
from transformers import Trainer, TrainingArguments  # noqa: F401
print("HF FT env ready — next: sbatch slurm/ft_gd_train.sh")
PY

echo
echo "============================================================"
echo "HF GroundingDINO FT env ready in .gdino_venv."
echo "  build dataset (in .sam3_venv):  python -m tools.ft_dataset_convert ..."
echo "  train:                          sbatch slurm/ft_gd_train.sh"
echo "============================================================"
