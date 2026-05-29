#!/bin/bash
#SBATCH --job-name=ocr_slots_paddle_cpu
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-37%38
#SBATCH --output=logs/ocr_slots_paddle_cpu_%A_%a.out
#SBATCH --error=logs/ocr_slots_paddle_cpu_%A_%a.err
#
# CPU-ONLY paddle slot-OCR. PaddleOCR defaults to CPU when --use-gpu is not
# passed. cpu_short has order-of-magnitude more capacity than the GPU
# partitions, so all videos run at once (array throttle %38 = no throttle).
#
# ENV: sources .sam3_venv (paddleocr 3.5.0 + rapidfuzz). The old
# /gpfs/.../.paddle_venv segfaults on `import paddleocr` and must NOT be used.
# Paddle models must be pre-cached (login node has internet; compute nodes may
# not) — warm ~/.paddlex once before submitting.
#
# SINGLE-VIDEO TEST:  sbatch --array=0 slurm/ocr_slots_paddle_cpu.sh
# FULL COHORT:        sbatch slurm/ocr_slots_paddle_cpu.sh

set -euo pipefail
REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs

source "$REPO/.sam3_venv/bin/activate"

# PaddlePaddle 3.x CPU bug: the PIR new-executor's oneDNN instruction path
# throws "ConvertPirAttribute2RuntimeAttribute not support" on every OCR call.
# Disable oneDNN + the PIR executor so CPU OCR falls back to stable kernels.
export FLAGS_use_mkldnn=0
export FLAGS_enable_pir_in_executor=0

python -c "import rapidfuzz" 2>/dev/null || { echo "FATAL: rapidfuzz missing in .sam3_venv"; exit 3; }

FRAMES_ROOT=/gpfs/data/oermannlab/private_data/whip/frames_attempt2

VIDEOS=(
  DC_whip_11609423 DG_whip_16598313 DOS_whip_15265421 DP_whip_16005154
  DS_whip_16931084 EG_whip_13471773 EP_SMC_whip_11902050 ER_whip_1596672
  GA_whip_0657679 HM_whip_12096998 JB_whip_1477332 JL_whip_13970873
  JM_compP_15397614 JP_whip_liver_5151473 JS_whip_12110018 LJ_whip_0725847
  LK_whip_16604333 MF_whip_10662561 MF_whip_12259774 MT_whip_0925905
  NP_whip_16798242 PT_whip_9397863 RE_whip_8825001 RG_whip_0705734
  RR_whip_15132384 RR_whip_16769698 RS_whip_16759779 RV_whip_11406119
  SD_whip_16926732 SK_whip_1374572 SP_whip_11877572 SR_whip_9631270
  SS_whip_5084312 SS_whip_9302344 UP_whip_2132042 VK_whip_11947036
  VK_whip_14621770 YMW_whip_16579351
)
VID="${VIDEOS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task $SLURM_ARRAY_TASK_ID -> $VID on $(hostname) (CPU) ==="
echo "cpus: ${SLURM_CPUS_PER_TASK:-?}"

# Hint paddle/numpy to use the slurm-allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

python tools/ocr_slots_timeline_paddle.py \
    --video-id "$VID" \
    --frames-dir "$FRAMES_ROOT/$VID" \
    --out-dir results/ocr_slots_timeline_paddle_full \
    --queries-file configs/cardiac_whip_vocabulary.json \
    --stride 30 \
    --upscale 3.0 \
    --fuzz-threshold 55

echo "task $SLURM_ARRAY_TASK_ID ($VID) done at $(date -Iseconds)"
