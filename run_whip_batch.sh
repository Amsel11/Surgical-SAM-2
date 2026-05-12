#!/bin/bash
#SBATCH --job-name=surgsam2_whip
#SBATCH --partition=oermannlab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/surgsam2_%j.out
#SBATCH --error=logs/surgsam2_%j.err
# Submit with:  sbatch run_whip_batch.sh
# Logs go to logs/surgsam2_<jobid>.out
#
# Runs multi-instrument SurgSAM-2 inference on the whip videos listed below.
# Comment / uncomment entries to choose which videos to process.

set -euo pipefail
mkdir -p logs

# --- environment ---
# Option A (current state on bigpurple, 2026-05-12): conda env `surgsam` exists
# but torch is NOT installed in it. So we create a uv venv on first run.
REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"

if [ ! -d .venv ]; then
    echo "Bootstrapping .venv with uv (one-time, ~3 min)"
    module load python/cpu/3.11.4 2>/dev/null || true
    uv venv --python 3.11 .venv
    uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    SAM2_BUILD_CUDA=0 uv pip install -e ".[notebooks]"
    uv pip install imageio-ffmpeg pandas scikit-image
fi
source .venv/bin/activate

# --- data ---
FRAMES_ROOT=/gpfs/data/oermannlab/private_data/whip/frames_attempt2
RESULTS_ROOT="$REPO/results"

# --- video list (uncomment to enable) ---
# Format: VIDEO_NAME "OBJID:x,y[ OBJID:x,y ...]"
# Pick clicks by looking at frame 0 of each video first; the (540,150) and
# (280,180) defaults are calibrated only for DC_whip_11609423.
declare -a VIDEOS=(
    "DC_whip_11609423|1:540,150 2:280,180"
    # "DG_whip_16598313|1:600,200 2:300,250"
    # "DP_whip_16005154|1:500,180 2:250,200"
    # "JS_whip_12110018|1:550,170 2:270,220"
)

for entry in "${VIDEOS[@]}"; do
    vid="${entry%%|*}"
    prompts="${entry##*|}"
    frames_dir="$FRAMES_ROOT/$vid"
    out_dir="$RESULTS_ROOT/${vid}_both"
    if [ ! -d "$frames_dir" ]; then
        echo "SKIP $vid: $frames_dir not found"
        continue
    fi
    echo "=== Running on $vid ==="
    obj_args=""
    for p in $prompts; do
        obj_args+="--object $p "
    done
    python run_on_video.py \
        --video "$frames_dir" \
        --output-dir "$out_dir" \
        --fps 1.0 \
        $obj_args
    echo "  -> $out_dir/overlay.mp4"
done

echo "All done."
