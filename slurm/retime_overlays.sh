#!/bin/bash
#SBATCH --job-name=retime_ov
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/retime_ov_%j.out
#SBATCH --error=logs/retime_ov_%j.err
#
# Re-time every overlay.mp4 from its current (buggy) r_frame_rate=1 to the
# correct 6 fps. The pipeline encoded overlays at the manifest's videos.fps
# default (1.0) instead of our tracking fps (6.0), stretching every overlay
# 6x in playback duration. Fix: ffmpeg -itsscale 0.16666667 + -c copy --
# instant container-level retiming, NO re-encode, NO content loss.
#
# Atomic replace: write to .tmp then mv. Skips a clip if log.json doesn't
# exist (run still in progress) or if the overlay is newer than its log
# (still being written).
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
    results/final_pja/pja_auto_sam3_baseline_v1
    results/whip_auto_surgsam2_v1
)

n_ok=0; n_skip=0; n_fail=0
for root in "${ROOTS[@]}"; do
    [ -d "$root" ] || continue
    for ov in "$root"/*/seed_1/overlay.mp4; do
        [ -f "$ov" ] || continue
        log="$(dirname "$ov")/log.json"
        # Skip if the run isn't finished (no log.json) or overlay is being
        # written right now (overlay newer than log).
        if [ ! -f "$log" ] || [ "$ov" -nt "$log" ]; then
            n_skip=$(( n_skip + 1 )); continue
        fi
        # Check current rate -- if already 6/1, skip (idempotent).
        rate=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$ov" 2>/dev/null)
        if [ "$rate" = "6/1" ]; then
            n_skip=$(( n_skip + 1 )); continue
        fi
        tmp="${ov}.retime.tmp"
        if ffmpeg -nostdin -loglevel error -y -itsscale 0.16666667 -i "$ov" \
                -c copy "$tmp" 2>>logs/retime_ov_${SLURM_JOB_ID}.err && [ -s "$tmp" ]; then
            mv "$tmp" "$ov"
            n_ok=$(( n_ok + 1 ))
        else
            rm -f "$tmp"
            n_fail=$(( n_fail + 1 ))
            echo "FAIL: $ov"
        fi
    done
done
echo "=== retime done: $n_ok fixed / $n_skip skipped / $n_fail failed ==="
