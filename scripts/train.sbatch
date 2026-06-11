#!/usr/bin/env bash
#SBATCH --job-name=chesslm_train
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --partition=pli-c
#
# Minimal, config-driven launcher — all knobs live in the YAML, so this script
# is the same for every run; only the config differs:
#   sbatch scripts/train.sbatch configs/dryrun.yaml
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$REPO/configs/dryrun.yaml}"
VENV=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/.venv

source "$VENV/bin/activate"

# Make the checkout importable as the package name `chesslm` regardless of the
# directory name (our clone is `merged_code`).
PKG="$REPO/.runpkg"; mkdir -p "$PKG"; ln -sfn "$REPO" "$PKG/chesslm"
export PYTHONPATH="$PKG:${PYTHONPATH:-}"

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "launching: config=$CONFIG  ngpu=$NGPU"
torchrun --standalone --nproc_per_node="$NGPU" "$REPO/train.py" --config "$CONFIG"
