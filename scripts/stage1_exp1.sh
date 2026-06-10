#!/usr/bin/env bash
# Historical record of Experiment 1 (Stage 1 architecture comparison).
# NOT intended for rerunning. Results in chesslm/runs/stage1_*/; analysis in chesslm/exp1.md.
#
# Two sub-sweeps were submitted as separate SLURM array jobs:
#
#   Baseline (array=0-7):  lr=1e-4, constant schedule, --time=36:00:00
#   High-LR  (array=0-7):  lr=2e-4, cosine schedule,  --time=24:00:00, exp_name += _hilr
#
# Array layout (arch_idx * 2 + dataset_idx):
#   0  flamingo              v2.1
#   1  flamingo              v3
#   2  kv_proj/channel_concat  v2.1
#   3  kv_proj/channel_concat  v3
#   4  kv_proj/interleaved   v2.1
#   5  kv_proj/interleaved   v3
#   6  llava                 v2.1
#   7  llava                 v3

# ── Baseline job ────────────────────────────────────────────────────────────
# #SBATCH --job-name=chesslm-s1
# #SBATCH --time=36:00:00
# #SBATCH --array=0-7
# #SBATCH --output=chesslm/logs/array_%A_%a.out

# ── High-LR job ─────────────────────────────────────────────────────────────
# #SBATCH --job-name=chesslm-s1-hilr
# #SBATCH --time=24:00:00
# #SBATCH --array=0-7
# #SBATCH --output=chesslm/logs/array_hilr_%A_%a.out

# ── Shared SBATCH options (both jobs) ───────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=pli-c
#SBATCH --mail-type=fail
#SBATCH --mail-user=jc93@princeton.edu

set -xeuo pipefail

source /scratch/gpfs/DANQIC/jeff/chesslm/.venv/bin/activate

TASK_ID="${SLURM_ARRAY_TASK_ID}"

ARCH_IDX=$(( TASK_ID / 2 ))
DATASET_IDX=$(( TASK_ID % 2 ))

DATASET_VERSIONS=("v2.1" "v3")
DATASET_VERSION="${DATASET_VERSIONS[$DATASET_IDX]}"
DATASET_DIR="/scratch/gpfs/DANQIC/jeff/chesslm/chesslm/datasets/${DATASET_VERSION}"

case "$ARCH_IDX" in
    0)
        ARCH="flamingo"
        LORA_RANK=-1
        ARCH_EXTRA_ARGS=""
        EXP_TAG="flamingo"
        ;;
    1)
        ARCH="kv_proj"
        LORA_RANK=16
        ARCH_EXTRA_ARGS="--proj-mode channel_concat"
        EXP_TAG="kv_proj_channel_concat"
        ;;
    2)
        ARCH="kv_proj"
        LORA_RANK=16
        ARCH_EXTRA_ARGS="--proj-mode interleaved"
        EXP_TAG="kv_proj_interleaved"
        ;;
    3)
        ARCH="llava"
        LORA_RANK=16
        ARCH_EXTRA_ARGS=""
        EXP_TAG="llava"
        ;;
esac

# Baseline: LR=1e-4 constant, no _hilr suffix.
# High-LR:  LR=2e-4 cosine,   _hilr suffix, --lr 2e-4 --scheduler cosine appended.
EXP_NAME="stage1_${EXP_TAG}_${DATASET_VERSION}${HILR_SUFFIX:-}"

python -m chesslm.train \
    --arch              "$ARCH" \
    --lora-rank         "$LORA_RANK" \
    ${ARCH_EXTRA_ARGS} \
    --exp-name          "$EXP_NAME" \
    --output-dir        /scratch/gpfs/DANQIC/jeff/chesslm/chesslm/runs/ \
    --decoder-path      /scratch/gpfs/DANQIC/jeff/models/smollm-3b-instruct \
    --encoder-path      /scratch/gpfs/DANQIC/jeff/chesslm/chesslm/encoder/lc0_hf_bt5 \
    --train-dataset     "${DATASET_DIR}/train" \
    --eval-dataset      "${DATASET_DIR}/eval" \
    --batch-size        64 \
    --n-steps           50000 \
    --grad-accum-steps  4 \
    --eval-freq         5000 \
    --log-samples       64 \
    --num-workers       8 \
    ${LR_ARGS:-} \
    ${RESUME_FROM:+--resume-from "$RESUME_FROM"} \
    ${EXTRA_ARGS:-}
# Baseline invocation: sbatch chesslm/scripts/stage1_exp1.sh
# High-LR invocation:  HILR_SUFFIX=_hilr LR_ARGS="--lr 2e-4 --scheduler cosine" \
#                        sbatch chesslm/scripts/stage1_exp1.sh
