#!/usr/bin/env bash
# Dryrun smoketest for LLaVAChessLM training. Tests the full pipeline
# (init, train steps, eval, checkpoint save) with minimal compute.
# Run with:
#   TEST_MODE=1 srun --partition=pli-c --gres=gpu:1 --mem=64G --time=00:20:00 bash chesslm/scripts/dryruns/llava_dryrun.sh
set -xeuo pipefail

source /scratch/gpfs/DANQIC/jeff/chesslm/.venv/bin/activate

TEST_MODE=1 python -m chesslm.train \
    --arch                llava \
    --lora-rank           16 \
    --exp-name            llava_dryrun \
    --output-dir          chesslm/runs_dryrun/ \
    --decoder-path        /scratch/gpfs/DANQIC/jeff/models/smollm-3b-instruct \
    --encoder-path        /scratch/gpfs/DANQIC/jeff/chesslm/chesslm/encoder/lc0_hf_bt5 \
    --train-dataset       /scratch/gpfs/DANQIC/jeff/chesslm/chesslm/datasets/v2/train \
    --eval-dataset        /scratch/gpfs/DANQIC/jeff/chesslm/chesslm/datasets/v2/eval \
    --batch-size          64 \
    --n-steps             50 \
    --grad-accum-steps    4 \
    --eval-freq           25 \
    --eval-batch-size     64 \
    --eval-max-examples   64 \
    --log-samples         2 \
    --num-workers         4
