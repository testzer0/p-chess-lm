#!/usr/bin/env bash
# Quick smoketest for both probe types. Run with:
#   srun --partition=pli-c --gres=gpu:1 --mem=16G --time=00:10:00 bash chess/scripts/smoketest.sh
set -xeuo pipefail

source /scratch/gpfs/DANQIC/jeff/chesslm/.venv/bin/activate

COMMON_ARGS=(
    --jsonl chess/data/positions.jsonl
    --lc0-weights chess/encoder/lc0_hf_bt5
    --layer-idx 8
    --max-steps 500
    --log-every 50
    --eval-every 100
    --num-workers 2
)

python chess/probe.py "${COMMON_ARGS[@]}" \
    --probe-type piece \
    --out-dir chess/piece_dryrun

python chess/probe.py "${COMMON_ARGS[@]}" \
    --probe-type attack \
    --out-dir chess/attack_dryrun
