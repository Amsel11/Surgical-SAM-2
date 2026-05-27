#!/bin/bash
#SBATCH --job-name=gdino_setup
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gdino_setup_%j.out
#SBATCH --error=logs/gdino_setup_%j.err
#
# One-shot setup for the Grounding DINO Stage-1 prompter on bp. Builds a
# separate venv (.gdino_venv) so the sam2/sam3 envs stay clean — the
# transformers pins required by GD conflict with what SAM ships with.
#
# Submit:  sbatch slurm/setup_gdino_bp.sh
#
# After it finishes, do the first prompter run interactively so HuggingFace
# can warm its cache + download the ~1.3 GB grounding-dino-tiny weights:
#
#   ssh bp
#   cd /gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
#   source .gdino_venv/bin/activate
#   python -m tools.build_vocab               # if not done already
#   python -m tools.run_dino_prompter --video DC_whip_11609423 --seed 1
#
# Then, in .sam3_venv, run SAM 3 against the dino prompts:
#   python -m pipeline run +experiment=sam3_oob_whip \
#       stage1_prompting.method=dino "scope.videos=[DC_whip_11609423]"
#
# (The dino prompt_sets row inserted by run_dino_prompter is picked up by
# the orchestrator's _resolve_dino path — no on-demand generation needed.)

set -euo pipefail

REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs

# ── 1. Build venv ─────────────────────────────────────────────────────
if [ ! -d .gdino_venv ]; then
    PY="$(command -v python3.11 || command -v python3)"
    echo "Building venv at .gdino_venv with $PY"
    "$PY" -m venv .gdino_venv
else
    echo ".gdino_venv already exists; reusing"
fi

source .gdino_venv/bin/activate
python -m pip install --upgrade pip

# ── 2. Install runtime deps ───────────────────────────────────────────
# GD-tiny on HF needs transformers >= 4.40; we pin >= 4.51 to use the
# renamed post_process kwargs (the detector wrapper handles both via the
# try/except shim, but newer is friendlier).
pip install "torch>=2.4" "transformers>=4.51" "pillow>=10" "numpy<3"

# Pipeline glue. python-dotenv lets the tool scripts read .env for paths
# like SURGSAM_MANIFEST. pydantic+omegaconf are imported via pipeline.config
# if anything in this env ever needs to instantiate the full PipelineConfig
# (cheap, harmless to install).
pip install "pydantic>=2.0" "omegaconf>=2.3" python-dotenv

# ── 3. Smoke-test imports ─────────────────────────────────────────────
python - <<'PY'
from pipeline.prompts._grounding_dino_detector import (
    GroundingDinoConfig,
    GroundingDinoDetector,
)
from pipeline.prompts.dino import GroundingDinoPrompter, load_vocab
print(".gdino_venv imports clean — no SAM deps required")
print("ready for: python -m tools.run_dino_prompter --video <id>")
PY

echo
echo "============================================================"
echo "Setup complete. Smoke-test path:"
echo "  1.  ssh bp && cd $REPO && source .gdino_venv/bin/activate"
echo "  2.  python -m tools.audit_labels --cohort whip      # confirm labels populated"
echo "  3.  python -m tools.build_vocab                      # writes configs/cardiac_whip_vocab.json"
echo "  4.  python -m tools.run_dino_prompter --video <VID>  # populates prompt_sets"
echo "  5.  source .sam3_venv/bin/activate"
echo "  6.  python -m pipeline run +experiment=sam3_oob_whip \\"
echo "          stage1_prompting.method=dino 'scope.videos=[<VID>]'"
echo "============================================================"
