#!/bin/bash
#SBATCH --job-name=sam3_setup
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sam3_setup_%j.out
#SBATCH --error=logs/sam3_setup_%j.err
#
# One-shot setup for SAM 3 on bp. Clones the repo, builds a separate venv
# so the existing sam2 env stays clean, and installs the package.
# Run via:  sbatch slurm/setup_sam3_bp.sh
#
# After this finishes, log in interactively once to authenticate HF and
# download the checkpoint:
#
#   ssh bp
#   cd /gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
#   source .sam3_venv/bin/activate
#   hf auth login        # paste HF token (https://huggingface.co/settings/tokens)
#   huggingface-cli download facebook/sam3.1 \
#       --local-dir checkpoints/sam3.1 \
#       --include "*.pt" "*.pth" "*.bpe" "*.json"
#
# Then the pipeline can run:
#   python -m pipeline run +experiment=sam3_oob_whip

set -euo pipefail

REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs

# ── 1. Clone the SAM 3 repo (idempotent) ──────────────────────────────
if [ ! -d sam3 ]; then
    echo "Cloning facebookresearch/sam3 ..."
    git clone https://github.com/facebookresearch/sam3.git
else
    echo "sam3/ already exists; pulling latest"
    (cd sam3 && git pull --ff-only || true)
fi

# ── 2. Separate venv (don't pollute the sam2 .venv) ───────────────────
if [ ! -d .sam3_venv ]; then
    # Use Python 3.11 if available; SAM 3 supports modern Python.
    PY="$(command -v python3.11 || command -v python3)"
    echo "Building venv at .sam3_venv with $PY"
    "$PY" -m venv .sam3_venv
fi

source .sam3_venv/bin/activate
python -m pip install --upgrade pip

# ── 3. Install SAM 3 + helpers ────────────────────────────────────────
pip install -e ./sam3
# huggingface-cli + token-based auth helper
pip install "huggingface_hub[cli]"
# Match the sam2 env basics so downstream IO helpers work.
pip install opencv-python pillow imageio imageio-ffmpeg

# ── 4. Smoke test that the imports resolve ────────────────────────────
python - <<'PY'
import sam3
from sam3.model_builder import build_sam3_video_model
print(f"SAM 3 module loaded: {sam3.__file__}")
print(f"build_sam3_video_model OK")
PY

echo
echo "============================================================"
echo "Setup complete. Next manual steps (interactive):"
echo "  1.  ssh bp"
echo "  2.  cd $REPO && source .sam3_venv/bin/activate"
echo "  3.  hf auth login                  # paste HF token"
echo "  4.  huggingface-cli download facebook/sam3.1 \\"
echo "        --local-dir checkpoints/sam3.1"
echo "  5.  ls -la checkpoints/sam3.1/     # confirm .pth file present"
echo "  6.  Update configs/experiment/sam3_oob_whip.yaml's checkpoint:"
echo "      path to the actual filename from step 5."
echo "============================================================"
