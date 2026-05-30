#!/bin/bash
#SBATCH --job-name=pja_sam2
#SBATCH --partition=superpod
#SBATCH --nodelist=sp-0010
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --mem=1900G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pja_sam2_%j.out
#SBATCH --error=logs/pja_sam2_%j.err
#
# Same packed driver as auto_track_pja_pack_sp0010.sh, but Stage-2 is
# VANILLA SAM 2.1 hiera-small (not SurgSAM-2). Composes
# +experiment=sam2_oob_whip + stage1=auto + auto-extracted segments. Results
# land under results/final_pja/pja_auto_sam2_v1/<clip>/seed_1/ (won't clobber
# the SurgSAM-2 results we already have).
set -uo pipefail
REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"; mkdir -p logs
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

if [ -n "${CLIPS:-}" ]; then
    read -r -a STEMS <<< "$CLIPS"
else
    mapfile -t STEMS < <(ls "$OCRROOT"/*/segments.csv 2>/dev/null \
        | sed -E 's#.*/([^/]+)/segments\.csv$#\1#' | sort)
fi
echo "=== PJA SAM2-VANILLA auto-track: ${#STEMS[@]} clips, FPS=$FPS, CONC=$CONC @ $(date -Iseconds) ==="

run_one () {
    local stem="$1" gpu="$2"
    local scr="$SCRBASE/pja_sam2_$stem"
    rm -rf "$scr"; mkdir -p "$scr/frames/$stem" "$scr/seg/$stem"
    local clip="$CLIPS_DIR/$stem.mp4"
    [ -f "$clip" ] || { echo "MISSING $clip"; return 1; }
    [ -f "$OCRROOT/$stem/segments.csv" ] || { echo "MISSING OCR $stem"; return 1; }

    ffmpeg -nostdin -loglevel error -i "$clip" -vf "fps=$FPS" \
        "$scr/frames/$stem/frame_%06d.jpg" || return 1
    local nf; nf=$(ls "$scr/frames/$stem"/*.jpg 2>/dev/null | wc -l)
    [ "$nf" -gt 0 ] || { echo "[$stem] zero frames"; return 1; }
    python -m tools.rescale_segments --in "$OCRROOT/$stem/segments.csv" \
        --out "$scr/seg/$stem/segments.csv" --fps "$FPS" || return 1

    export SURGSAM_MANIFEST="$scr/manifest.db"
    python -m pipeline.cli scan-videos "$scr/frames" --cohort pja >/dev/null 2>&1 || return 1
    sqlite3 "$scr/manifest.db" "ATTACH DATABASE '$REPO/manifest.db' AS canonical; \
        INSERT OR IGNORE INTO instruments SELECT * FROM canonical.instruments; \
        DETACH DATABASE canonical;" || return 1

    export BP_RESULTS_ROOT="$REPO/results/final_pja"
    CUDA_VISIBLE_DEVICES="$gpu" python -m pipeline run \
        +experiment=sam2_oob_whip \
        name=pja_auto_sam2_v1 \
        stage1_prompting.method=auto \
        "scope.videos=[$stem]" \
        scope.seed=1 \
        "stage1_prompting.segments_root=$scr/seg"
    local rc=$?
    rm -rf "$scr/frames"
    echo "[$stem] rc=$rc @ $(date -Iseconds)"
    return $rc
}

i=0
for stem in "${STEMS[@]}"; do
    gpu=$(( i % NGPU ))
    run_one "$stem" "$gpu" > "logs/pja_sam2_${SLURM_JOB_ID}_${stem}.log" 2>&1 &
    i=$(( i + 1 ))
    while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 5; done
done
wait
echo "=== all ${#STEMS[@]} PJA-SAM2 clips done @ $(date -Iseconds) ==="
