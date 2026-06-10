#!/usr/bin/env bash
#SBATCH --job-name=piece-probe-balanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=pli-c
#SBATCH --array=0,3,7,11,15
#SBATCH --output=chess/logs/piece_probe_balanced_%A_%a.out
#SBATCH --mail-type=fail
#SBATCH --mail-user=jc93@princeton.edu

set -xeuo pipefail

source /scratch/gpfs/DANQIC/jeff/chesslm/.venv/bin/activate

python chess/probe.py \
    --jsonl chess/data/positions.jsonl \
    --lc0-weights chess/encoder/lc0_hf_bt5 \
    --layer-idx "$SLURM_ARRAY_TASK_ID" \
    --probe-type piece \
    --run-name piece_balanced \
    --max-steps 15000 \
    --out-dir chess/probe_outputs
