#!/bin/bash
#SBATCH --job-name=quick_ov
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/quick_ov_%j.out
#SBATCH --error=logs/quick_ov_%j.err
#
# For each <variant>/<clip>/seed_1/overlay.mp4 produced by the auto-track runs,
# render a small fast scroll-preview quick.mp4 next to it:
#   - scaled to 640 wide
#   - sped up 4x  (PTS/4)
#   - capped at 12 fps
#   - libx264 veryfast / CRF 28  (~2-5 MB per file)
# Idempotent: skips files whose quick.mp4 is newer than the source overlay.
#
# Submit once; covers every variant under final_pja/ + the whip pilot.
set -uo pipefail
REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
export PATH=/gpfs/share/apps/ffmpeg/7.1.1/bin:$PATH
command -v ffmpeg >/dev/null || { echo "FATAL: ffmpeg not on PATH"; exit 2; }

ROOTS=(
    results/final_pja/pja_auto_surgsam2_v1
    results/final_pja/pja_auto_surgsam2_baseline_v1
    results/final_pja/pja_auto_surgsam2_dense300_v1
    results/final_pja/pja_auto_surgsam2_dense600_v1
    results/final_pja/pja_auto_surgsam2_dense1200_v1
    results/final_pja/pja_auto_sam2_v1
    results/whip_auto_surgsam2_v1
)

n_made=0; n_skip=0; n_fail=0
t0=$(date +%s)
for root in "${ROOTS[@]}"; do
    [ -d "$root" ] || { echo "skip (no dir): $root"; continue; }
    echo "=== $root ==="
    for ov in "$root"/*/seed_1/overlay.mp4; do
        [ -f "$ov" ] || continue
        # quick_v2.mp4 = the playable rebuild. Previous quick.mp4 files were
        # encoded without -pix_fmt yuv420p and render all-green in QuickTime/
        # Safari/Photos. The bad ones can be ignored / deleted manually.
        out="$(dirname "$ov")/quick_v2.mp4"
        if [ -f "$out" ] && [ "$out" -nt "$ov" ]; then
            n_skip=$(( n_skip + 1 )); continue
        fi
        # bp's ffmpeg/7.1.1 has no libx264; mpeg4 is the only software encoder.
        # -pix_fmt yuv420p forces the standard chroma format so QuickTime /
        # Safari / Photos can decode (without it, mpeg4 may emit a chroma
        # format those players render as all-green). -q:v 5 is more reliable
        # than -b:v for mpeg4 quality control.
        if ffmpeg -nostdin -loglevel error -y -i "$ov" \
                -vf "scale=480:-2,setpts=PTS/4,fps=8,format=yuv420p" -an \
                -c:v mpeg4 -pix_fmt yuv420p -q:v 5 \
                "$out" 2>>logs/quick_ov_${SLURM_JOB_ID}.err; then
            n_made=$(( n_made + 1 ))
            if [ $(( n_made % 10 )) -eq 0 ]; then
                echo "  ... $n_made made @ $(( $(date +%s) - t0 ))s"
            fi
        else
            n_fail=$(( n_fail + 1 ))
            echo "  FAIL: $ov"
        fi
    done
done
echo "=== quick.mp4: $n_made made / $n_skip up-to-date / $n_fail failed @ $(( $(date +%s) - t0 ))s ==="
