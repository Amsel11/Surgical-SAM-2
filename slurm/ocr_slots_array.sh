#!/bin/bash
#SBATCH --job-name=ocr_slots
#SBATCH --partition=superpod
#SBATCH --nodelist=sp-0012
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-37%8
#SBATCH --output=logs/ocr_slots_%A_%a.out
#SBATCH --error=logs/ocr_slots_%A_%a.err
#
# Production: spatial per-slot OCR presence-timeline over all 38 whip videos.
# Resolution-aware slot crops + camera/empty labels. One video per task,
# 8 concurrent on sp-0012's 8 H100s.
set -euo pipefail
REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs
source .sam3_venv/bin/activate
export HF_HOME="$REPO/.hf_cache"
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import qwen_vl_utils" 2>/dev/null || pip install --quiet "qwen-vl-utils[decord]"

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
echo "=== task $SLURM_ARRAY_TASK_ID -> $VID on $(hostname) ==="

python tools/ocr_slots_timeline.py \
    --video-id "$VID" \
    --frames-dir "$FRAMES_ROOT/$VID" \
    --out-dir results/ocr_slots_timeline \
    --queries-file configs/cardiac_whip_vocabulary.json \
    --stride 30

echo "task $SLURM_ARRAY_TASK_ID ($VID) done at $(date -Iseconds)"
