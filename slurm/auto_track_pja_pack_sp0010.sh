#!/bin/bash
#SBATCH --job-name=pja_auto
#SBATCH --partition=superpod
#SBATCH --nodelist=sp-0010
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --mem=1900G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pja_auto_%j.out
#SBATCH --error=logs/pja_auto_%j.err
#
# ZERO-SHOT auto-track for the 85-clip PJA FINAL cohort, packed on SP-0010
# (8 H100 / 224 cores / 2TB). Approach B: per clip, ffmpeg-extract at tracking
# fps to node-local scratch, rescale the 1-fps OCR segments to that fps, build a
# throwaway temp manifest, run auto(GD)->SurgSAM-2->dedup, persist results to
# results/final_pja/, discard scratch. No permanent frame store, no shared-
# manifest contention (each clip gets its own temp db).
#
# Tunables (override via --export=ALL,FPS=6,CONC=16,CLIPS="A B"):
#   FPS   tracking-frame fps for ffmpeg + segment rescale (default 6)
#   CONC  concurrent clips across the 8 GPUs (RAM-bound, default 16)
#   CLIPS space-separated stem list (default: all clips that HAVE OCR segments)
#
#   Validate one:  sbatch --export=ALL,CLIPS=AB_whip_15995945_PJA slurm/auto_track_pja_pack_sp0010.sh
#   Full cohort:   sbatch slurm/auto_track_pja_pack_sp0010.sh
set -uo pipefail
REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"; mkdir -p logs
# ffmpeg lives on shared GPFS; bypass the module system (superpod compute
# nodes don't pick up the modulefiles path reliably).
export PATH=/gpfs/share/apps/ffmpeg/7.1.1/bin:$PATH
source .sam3_venv/bin/activate
command -v ffmpeg >/dev/null || { echo "FATAL: ffmpeg not at /gpfs/share/apps/ffmpeg/7.1.1/bin"; exit 2; }

export BP_REPO="$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

CLIPS_DIR=/gpfs/data/oermannlab/users/schula12/whipple-transfer/clips
OCRROOT="$REPO/results/ocr_paddle_clips"
FPS="${FPS:-6}"
CONC="${CONC:-16}"
NGPU=8
SCRBASE="${SLURM_TMPDIR:-/tmp}"

# Clip list: explicit CLIPS, else every clip that already has OCR segments.
if [ -n "${CLIPS:-}" ]; then
    read -r -a STEMS <<< "$CLIPS"
else
    mapfile -t STEMS < <(ls "$OCRROOT"/*/segments.csv 2>/dev/null \
        | sed -E 's#.*/([^/]+)/segments\.csv$#\1#' | sort)
fi
echo "=== PJA auto-track: ${#STEMS[@]} clips, FPS=$FPS, CONC=$CONC on $(hostname) @ $(date -Iseconds) ==="

run_one () {
    local stem="$1" gpu="$2"
    local scr="$SCRBASE/pja_$stem"
    rm -rf "$scr"; mkdir -p "$scr/frames" "$scr/seg/$stem"
    local clip="$CLIPS_DIR/$stem.mp4"
    [ -f "$clip" ] || { echo "MISSING clip $clip"; return 1; }
    [ -f "$OCRROOT/$stem/segments.csv" ] || { echo "MISSING OCR $stem"; return 1; }

    echo "[$stem] ffmpeg fps=$FPS @ $(date -Iseconds)"
    ffmpeg -nostdin -loglevel error -i "$clip" -vf "fps=$FPS" \
        "$scr/frames/$stem/frame_%06d.jpg" || { echo "[$stem] ffmpeg FAIL"; return 1; }
    local nf; nf=$(ls "$scr/frames/$stem"/*.jpg 2>/dev/null | wc -l)
    echo "[$stem] extracted $nf frames"
    [ "$nf" -gt 0 ] || { echo "[$stem] zero frames"; return 1; }

    python -m tools.rescale_segments --in "$OCRROOT/$stem/segments.csv" \
        --out "$scr/seg/$stem/segments.csv" --fps "$FPS" || return 1

    export SURGSAM_MANIFEST="$scr/manifest.db"
    python -m pipeline.cli scan-videos "$scr/frames" --cohort pja >/dev/null 2>&1 || return 1

    export BP_RESULTS_ROOT="$REPO/results/final_pja"
    CUDA_VISIBLE_DEVICES="$gpu" python -m pipeline run \
        +experiment=whip_auto_surgsam2 \
        name=pja_auto_surgsam2_v1 \
        "scope.videos=[$stem]" \
        scope.seed=1 \
        "stage1_prompting.segments_root=$scr/seg"
    local rc=$?
    rm -rf "$scr/frames"   # free node-local scratch promptly
    echo "[$stem] rc=$rc @ $(date -Iseconds)"
    return $rc
}

i=0
for stem in "${STEMS[@]}"; do
    gpu=$(( i % NGPU ))
    run_one "$stem" "$gpu" > "logs/pja_auto_${SLURM_JOB_ID}_${stem}.log" 2>&1 &
    echo "launched [$i] $stem -> GPU $gpu (pid $!)"
    i=$(( i + 1 ))
    while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 5; done
done
wait
echo "=== all ${#STEMS[@]} PJA clips done @ $(date -Iseconds) ==="
ok=0; fail=0
for stem in "${STEMS[@]}"; do
    if grep -q 'rc=0' "logs/pja_auto_${SLURM_JOB_ID}_${stem}.log" 2>/dev/null; then ok=$((ok+1)); else fail=$((fail+1)); echo "FLAG: $stem"; fi
done
echo "=== ok=$ok fail=$fail of ${#STEMS[@]} ==="
