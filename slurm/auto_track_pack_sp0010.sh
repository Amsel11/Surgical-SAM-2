#!/bin/bash
#SBATCH --job-name=auto_pack
#SBATCH --partition=superpod
#SBATCH --nodelist=sp-0010
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --mem=1800G
#SBATCH --time=04:00:00
#SBATCH --output=logs/auto_pack_%j.out
#SBATCH --error=logs/auto_pack_%j.err
#
# PACKED zero-shot run: one allocation grabs the whole SP-0010 node (8 H100,
# 224 cores, ~2TB RAM) and runs many videos CONCURRENTLY, round-robining each
# across the 8 GPUs. The pipeline is CPU/IO-bound (~5% GPU), so packing
# several videos per GPU is the real throughput lever.
#
# Concurrency (CONC) is bounded by CPU cores + RAM, NOT GPUs. Default 24
# (~8 cores + plenty of RAM each). Bump via:  sbatch --export=CONC=32 ...
#
# Each video logs to logs/auto_pack_<jobid>_<video>.log
#
#   sbatch slurm/auto_track_pack_sp0010.sh            # all videos w/ OCR segments
#   sbatch --export=ALL,CONC=32 slurm/auto_track_pack_sp0010.sh
set -uo pipefail
REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"; mkdir -p logs
source .sam3_venv/bin/activate

export BP_REPO="$REPO"
export BP_FRAMES_ROOT=/gpfs/data/oermannlab/private_data/whip/frames_attempt2
export BP_RESULTS_ROOT="$REPO/results"
export SURGSAM_MANIFEST="$REPO/manifest.db"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep each process from grabbing all 224 cores (would thrash under packing).
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

SEGROOT="$REPO/results/ocr_slots_timeline_paddle_full"
NGPU=8
CONC="${CONC:-24}"

mapfile -t VIDEOS < <(ls "$SEGROOT"/*/segments.csv 2>/dev/null \
    | sed -E 's#.*/([^/]+)/segments\.csv$#\1#' | sort)
echo "=== PACKED auto-track: ${#VIDEOS[@]} videos, CONC=$CONC, ${NGPU} GPUs on $(hostname) @ $(date -Iseconds) ==="

i=0
for VID in "${VIDEOS[@]}"; do
    gpu=$(( i % NGPU ))
    CUDA_VISIBLE_DEVICES=$gpu python -m pipeline run \
        +experiment=whip_auto_surgsam2 \
        "scope.videos=[$VID]" \
        scope.seed=1 \
        "stage1_prompting.segments_root=$SEGROOT" \
        > "logs/auto_pack_${SLURM_JOB_ID}_${VID}.log" 2>&1 &
    echo "launched [$i] $VID -> GPU $gpu (pid $!)"
    i=$(( i + 1 ))
    # Concurrency gate: wait while >= CONC background jobs are running.
    while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 5; done
done
wait
echo "=== ALL ${#VIDEOS[@]} videos done @ $(date -Iseconds) ==="
# Per-video exit status summary.
fail=0
for VID in "${VIDEOS[@]}"; do
    if grep -qiE 'Traceback|Error' "logs/auto_pack_${SLURM_JOB_ID}_${VID}.log" 2>/dev/null; then
        echo "WARN possible failure: $VID"; fail=$(( fail + 1 ))
    fi
done
echo "=== $fail/${#VIDEOS[@]} videos flagged with errors in logs ==="
