#!/usr/bin/env bash
# Experiment 2 — Stage 1 architecture & LR sweep.
# 30 array jobs: 5 groups × (2 datasets × 3 LRs).
#
# Array layout (group_idx * 6 + dataset_idx * 3 + lr_idx):
#
#   Group 0 ( 0– 5): flamingo, frozen,       α=2.0 W_O=0
#   Group 1 ( 6–11): flamingo, lora16_open,  α=2.0 W_O=0,   --decoder-lr 2e-4
#   Group 2 (12–17): flamingo, lora16_std,   α=0.0 W_O=rand, --decoder-lr 2e-4
#   Group 3 (18–23): llava,    frozen
#   Group 4 (24–29): llava,    lora16,                        --decoder-lr 2e-4
#
# Within each group of 6:
#   position 0–2: v2.1  ×  lr 2e-4, 4e-4, 8e-4
#   position 3–5: v3    ×  lr 2e-4, 4e-4, 8e-4

#SBATCH --job-name=chesslm-s1-exp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=pli-c
#SBATCH --array=0-29
#SBATCH --chdir=/scratch/gpfs/DANQIC/jeff/chesslm
#SBATCH --output=/scratch/gpfs/DANQIC/jeff/chesslm/chesslm/logs/exp2_%A_%a.out
#SBATCH --mail-type=fail
#SBATCH --mail-user=jc93@princeton.edu

set -xeuo pipefail

source /scratch/gpfs/DANQIC/jeff/chesslm/.venv/bin/activate

TASK_ID="${SLURM_ARRAY_TASK_ID}"
GROUP_IDX=$(( TASK_ID / 6 ))
POS=$(( TASK_ID % 6 ))
DATASET_IDX=$(( POS / 3 ))
LR_IDX=$(( POS % 3 ))

DATASET_VERSIONS=("v2.1" "v3")
DATASET_VERSION="${DATASET_VERSIONS[$DATASET_IDX]}"
DATASET_DIR="/scratch/gpfs/DANQIC/jeff/chesslm/chesslm/datasets/arch_exps/${DATASET_VERSION}"

LRS=("2e-4" "4e-4" "8e-4")
LR_TAGS=("2e4" "4e4" "8e4")
LR="${LRS[$LR_IDX]}"
LR_TAG="${LR_TAGS[$LR_IDX]}"

EXTRA_ARGS=()

case "$GROUP_IDX" in
    0)
        ARCH="flamingo"
        LORA_RANK=-1
        EXP_TAG="flamingo_frozen"
        EXTRA_ARGS+=(--alpha-init 2.0 --wo-zero-init)
        ;;
    1)
        ARCH="flamingo"
        LORA_RANK=16
        EXP_TAG="flamingo_lora16_open"
        EXTRA_ARGS+=(--alpha-init 2.0 --wo-zero-init --decoder-lr 2e-4)
        ;;
    2)
        ARCH="flamingo"
        LORA_RANK=16
        EXP_TAG="flamingo_lora16_std"
        EXTRA_ARGS+=(--decoder-lr 2e-4)
        ;;
    3)
        ARCH="llava"
        LORA_RANK=-1
        EXP_TAG="llava_frozen"
        ;;
    4)
        ARCH="llava"
        LORA_RANK=16
        EXP_TAG="llava_lora16"
        EXTRA_ARGS+=(--decoder-lr 2e-4)
        ;;
esac

EXP_NAME="stage1_${EXP_TAG}_${DATASET_VERSION}_${LR_TAG}"

python -m train \
    --arch              "$ARCH" \
    --lora-rank         "$LORA_RANK" \
    --exp-name          "$EXP_NAME" \
    --output-dir        /scratch/gpfs/DANQIC/jeff/chesslm/chesslm/runs/arch_exp2/ \
    --decoder-path      /scratch/gpfs/DANQIC/jeff/models/smollm-3b-instruct \
    --encoder-path      /scratch/gpfs/DANQIC/jeff/chesslm/chesslm/encoder/lc0_hf_bt5 \
    --train-dataset     "${DATASET_DIR}/train" \
    --eval-dataset      "${DATASET_DIR}/eval" \
    --batch-size        64 \
    --n-steps           30000 \
    --grad-accum-steps  4 \
    --eval-freq         3000 \
    --log-samples       64 \
    --num-workers       8 \
    --lr                "$LR" \
    --scheduler         cosine \
    "${EXTRA_ARGS[@]}" \
    ${RESUME_FROM:+--resume-from "$RESUME_FROM"}
