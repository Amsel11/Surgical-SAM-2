#!/bin/bash
#SBATCH --job-name=surgsam2_jupyter
#SBATCH --partition=oermannlab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/jupyter_%j.out
#SBATCH --error=logs/jupyter_%j.err
#
# Submit with:  sbatch run_jupyter.sh
# Then:
#   1) tail -f logs/jupyter_<jobid>.out  -- look for the http://... URL with token
#   2) Note the compute node name (e.g. a100-2)  -- shown in the log header
#   3) From olab-1 set up the tunnel:
#        ssh -L 9000:localhost:18890 bigpurple \
#            "ssh -N -f -L 18890:localhost:8889 <NODE> && sleep infinity"
#   4) From your LAPTOP set up the outer tunnel:
#        ssh -L 9000:localhost:9000 schula12@olab-1
#   5) Open in browser:  http://127.0.0.1:9000/lab?token=<token>

set -euo pipefail
mkdir -p logs

REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"

# Bootstrap venv if missing
if [ ! -d .venv ]; then
    echo "Bootstrapping .venv with uv (one-time, ~3 min)"
    module load python/cpu/3.11.4 2>/dev/null || true
    if ! command -v uv >/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv venv --python 3.11 .venv
    uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    SAM2_BUILD_CUDA=0 uv pip install -e ".[notebooks]"
    uv pip install imageio-ffmpeg pandas scikit-image
fi
source .venv/bin/activate

echo "============================================================"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Port:    8889"
echo "Repo:    $REPO"
echo "============================================================"
echo
echo "Tunnel from olab-1:"
echo "  ssh -L 9000:localhost:18890 bigpurple \\"
echo "      \"ssh -N -f -L 18890:localhost:8889 $(hostname) && sleep infinity\""
echo "Tunnel from laptop:"
echo "  ssh -L 9000:localhost:9000 schula12@olab-1"
echo

jupyter lab --no-browser --ip=127.0.0.1 --port=8889 --ServerApp.open_browser=False
