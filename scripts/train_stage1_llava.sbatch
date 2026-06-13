#!/usr/bin/env bash
#SBATCH --job-name=train_stage1_llava
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --partition=pli-c
#
# Stage-1 LLaVA full-FT training (POV, 30k steps, global batch 256 on 2 GPUs).
#   sbatch scripts/train_stage1_llava.sbatch
# Checkpoints land in runs/stage1_llava_fullft_30k/ every eval_freq (2000) steps;
# to resume after a time-limit, add `resume_from: <ckpt_dir>` to the config or
# pass --resume-from on the torchrun line.
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
CONFIG="${1:-$REPO/configs/train_stage1_llava.yaml}"
VENV="$REPO/.venv"
source "$VENV/bin/activate"

# Importable as the package name `chesslm` regardless of checkout dir name.
PKG="$REPO/.runpkg"; mkdir -p "$PKG"; ln -sfn "$REPO" "$PKG/chesslm"
export PYTHONPATH="$PKG:${PYTHONPATH:-}"

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "launching: config=$CONFIG  ngpu=$NGPU"
torchrun --standalone --nproc_per_node="$NGPU" "$REPO/train.py" --config "$CONFIG"
