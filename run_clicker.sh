#!/bin/bash
#SBATCH --job-name=surgsam2_clicker
#SBATCH --partition=cpu_dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/clicker_%j.out
#SBATCH --error=logs/clicker_%j.err
#
# Gradio click collector on a CPU node. Reads/writes manifest.db on GPFS.
# Tunnel from laptop: bp_connect.sh pattern (clicker variant).
#
# Manual launch:
#   sbatch run_clicker.sh
#   # then check logs/clicker_<jobid>.out for the URL/port and node assignment

set -euo pipefail

REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs

if [ ! -d .venv ]; then
    echo "ERROR: .venv missing. Run run_inference_array.sh once to bootstrap it." >&2
    exit 1
fi
source .venv/bin/activate

# Ensure gradio is installed (idempotent)
python -c "import gradio" 2>/dev/null || uv pip install --quiet "gradio>=4.0,<5.0" 2>/dev/null || pip install --quiet "gradio>=4.0,<5.0"

PORT=${CLICKER_PORT:-9876}

echo "============================================================"
echo "Clicker on $(hostname)   port=$PORT"
echo "Manifest: $REPO/manifest.db"
echo
echo "Tunnel from laptop:"
echo "    ssh -L $PORT:localhost:$PORT bigpurple \"ssh -N -L $PORT:localhost:$PORT $(hostname)\""
echo "Then open http://localhost:$PORT in your browser."
echo "============================================================"

python -m pipeline.clicker --port "$PORT" --host 0.0.0.0
