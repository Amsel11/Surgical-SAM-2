#!/bin/bash
#SBATCH --job-name=surgsam2_array
#SBATCH --partition=oermannlab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#
# Inference over every prompts/<video>.json on bigpurple, one A100 per video.
#
# 1) Each prompts JSON corresponds to one video. The script enumerates all
#    prompts/*.json and the array index picks one of them.
# 2) Submit with the array size matching the JSON count:
#       N=$(ls prompts/*.json | wc -l)
#       sbatch --array=0-$((N-1))%8 run_inference_array.sh
#    (%8 caps concurrency at 8 nodes; remove the %8 to let SLURM run as many
#     as the partition allows.)
# 3) Per-video output goes to results/<video>/seed_1/. Add more seeds by
#    re-running with results/<video>/seed_<k>/ as output dir.
#
# Skips already-done videos (where results/<video>/seed_1/log.json exists).

set -euo pipefail

REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs

# --- Pick the prompts JSON for this array task ---
mapfile -t PJSONS < <(ls prompts/*.json 2>/dev/null | sort)
N=${#PJSONS[@]}
if [ $N -eq 0 ]; then
    echo "No prompts/*.json files found. Click first." ; exit 1
fi
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "$N" ]; then
    echo "SLURM_ARRAY_TASK_ID=$IDX is out of range (have $N prompts)" ; exit 1
fi
PJSON="${PJSONS[$IDX]}"

# Derive paths
BASENAME=$(basename "$PJSON" .json)              # e.g. DG_whip_16598313_seed1_manual_box
# Strip trailing _seed<N>_<method> if present, else use BASENAME as-is.
VIDEO_ID=$(echo "$BASENAME" | sed -E 's/_seed[0-9]+_[A-Za-z_]+$//')
FRAMES_DIR=/gpfs/data/oermannlab/private_data/whip/frames_attempt2/$VIDEO_ID
SEED=${SEED:-1}
OUT_DIR="$REPO/results/$VIDEO_ID/seed_$SEED"

# Idempotency
if [ -f "$OUT_DIR/log.json" ]; then
    echo "[$VIDEO_ID] seed_$SEED already done ($OUT_DIR/log.json exists) — skipping"
    exit 0
fi
if [ ! -d "$FRAMES_DIR" ]; then
    echo "[$VIDEO_ID] frames dir missing: $FRAMES_DIR" ; exit 1
fi

# --- Venv (bootstrap on first job ever; subsequent jobs just activate) ---
if [ ! -d .venv ]; then
    echo "Bootstrapping .venv (one-time)"
    module load python/cpu/3.11.4 2>/dev/null || true
    if ! command -v uv >/dev/null; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv venv --python 3.11 .venv
    uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    SAM2_BUILD_CUDA=0 uv pip install -e ".[notebooks]"
    uv pip install imageio-ffmpeg pandas scikit-image ipympl
fi
source .venv/bin/activate

# --- Run ---
echo "============================================================"
echo "[$VIDEO_ID] seed=$SEED  node=$(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Prompts: $PJSON"
echo "Frames:  $FRAMES_DIR"
echo "Output:  $OUT_DIR"
echo "============================================================"

mkdir -p "$OUT_DIR"
python run_on_video.py \
    --video "$FRAMES_DIR" \
    --prompts-json "$PJSON" \
    --output-dir "$OUT_DIR" \
    --fps 1.0 \
    --no-overlay-jpgs

echo "[$VIDEO_ID] done at $(date -Iseconds)"
