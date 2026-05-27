#!/bin/bash
#SBATCH --job-name=gdino_ft_setup
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gdino_ft_setup_%j.out
#SBATCH --error=logs/gdino_ft_setup_%j.err
#
# One-shot setup for Grounding DINO **fine-tuning** on bp via Open-GroundingDino.
# Separate from .gdino_venv (inference) because the FT repo pins different
# transformers / torch versions and runs its own training entrypoints.
#
# Open-GroundingDino is a community FT codepath for the IDEA-Research GD
# weights. HuggingFace transformers' GD wrapper has inference but no
# training, so this is the canonical way to LoRA-FT grounding-dino-tiny on a
# closed surgical vocabulary.
#
# Submit:  sbatch slurm/setup_gdino_ft_bp.sh
#
# After this finishes (first time), kick off the FT training with:
#   sbatch slurm/ft_gd_train.sh   (TODO)
#
# Caveat: the Open-GroundingDino repo's exact install command + dataset
# format (ODVG vs COCO) may have evolved since this script was written.
# If the smoke import at the bottom fails, check the repo's README for any
# new install steps and adjust here.

set -euo pipefail

REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs

# ── 1. Clone Open-GroundingDino (idempotent) ──────────────────────────
if [ ! -d Open-GroundingDino ]; then
    echo "Cloning longzw1997/Open-GroundingDino ..."
    git clone https://github.com/longzw1997/Open-GroundingDino.git
else
    echo "Open-GroundingDino/ already exists; pulling latest"
    (cd Open-GroundingDino && git pull --ff-only || true)
fi

# ── 2. Build separate venv (don't pollute .gdino_venv or .sam3_venv) ──
if [ ! -d .gdino_ft_venv ]; then
    PY="$(command -v python3.10 || command -v python3.11 || command -v python3)"
    echo "Building venv at .gdino_ft_venv with $PY"
    "$PY" -m venv .gdino_ft_venv
fi

source .gdino_ft_venv/bin/activate
python -m pip install --upgrade pip

# ── 3. Install Open-GroundingDino + deps ──────────────────────────────
# Open-GroundingDino's setup matches the IDEA-Research repo: torch +
# transformers + their own packaged ops. Their requirements.txt lives at
# the repo root.
cd Open-GroundingDino
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi
# Many of their ops live under groundingdino/models/GroundingDINO/ops with
# a setup.py for the CUDA extensions. Build them in-place.
if [ -f models/GroundingDINO/ops/setup.py ]; then
    (cd models/GroundingDINO/ops && python setup.py build install) || \
        echo "(non-fatal) CUDA ops build failed — may need to rerun on a GPU node"
fi

# LoRA support
pip install "peft>=0.5.0"

cd "$REPO"

# ── 4. Download the base SwinT checkpoint ─────────────────────────────
mkdir -p Open-GroundingDino/weights
if [ ! -f Open-GroundingDino/weights/groundingdino_swint_ogc.pth ]; then
    echo "Downloading groundingdino_swint_ogc.pth (~660 MB)..."
    wget -O Open-GroundingDino/weights/groundingdino_swint_ogc.pth \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "groundingdino_swint_ogc.pth already present"
fi

# ── 5. Smoke-test the import chain ────────────────────────────────────
cd Open-GroundingDino
python - <<'PY'
# Don't assume the exact module structure — print what's there.
import importlib, sys, pathlib
print(f"sys.path[0] = {sys.path[0]}")
print(f"contents:    {sorted(p.name for p in pathlib.Path('.').iterdir())[:20]}")
try:
    import groundingdino
    print(f"groundingdino package loaded from {groundingdino.__file__}")
except Exception as e:
    print(f"(non-fatal) groundingdino import: {e}")
PY
cd "$REPO"

echo
echo "============================================================"
echo "Setup complete. Next manual steps:"
echo "  1.  ssh bp && cd $REPO && source .gdino_ft_venv/bin/activate"
echo "  2.  python -m tools.ft_dataset_convert \\"
echo "          --in data/ft_labels_v1.jsonl \\"
echo "          --out Open-GroundingDino/data/whip_ft_v1/"
echo "  3.  sbatch slurm/ft_gd_train.sh    (writes checkpoints/gdino_whip_ft_v1.pth)"
echo
echo "If the CUDA-ops build at step 3 of this script failed, rerun it on"
echo "a GPU node:  srun --gres=gpu:1 --partition=a100_short --pty bash, then"
echo "  cd Open-GroundingDino/models/GroundingDINO/ops && python setup.py build install"
echo "============================================================"
