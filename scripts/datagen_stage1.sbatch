#!/bin/bash -l
#SBATCH --job-name=datagen_stage1
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=06:00:00
#
# Stage-1 curriculum datagen, end to end:
#   1. sample positions from the game-level split shards (parallel, 16-way),
#      n_per_game=5 to make ~12M positions available
#        train_*    -> data/lichess/train-NNN.jsonl
#        val_test_* -> data/lichess/val_test-000.jsonl
#   2. build the QA datasets via configs/datagen_stage1.yaml -> data/stage1/
#      (7 per-task train arrows + val + test, equal 1,828,572 records/task)
#   3. concatenate the 7 per-task train arrows into data/stage1/train_all.arrow
#      (equal sizes => uniform 1/7; DataLoader shuffles) for train.py.
#
# Usage:  sbatch scripts/datagen_stage1.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
PY="$REPO/.venv/bin/python"

ALLOC="$REPO/data/lichess_dbs/allocated/stage1"
OUT="$REPO/data/lichess"
NPG=5
mkdir -p "$OUT" "$REPO/joblog"

# Fresh start: previous run sampled n_per_game=1 (too few positions for the
# 12.8M-record budget). Remove old sampled shards and any prior build output.
rm -f "$OUT"/train-*.jsonl "$OUT"/val_test-*.jsonl
rm -rf "$REPO/data/stage1"

# --- 1. sampling -----------------------------------------------------------
sample_one() {
  local pgn="$1" out="$2"
  if [ -s "$out" ]; then echo "[skip] $out ($(wc -l < "$out") lines)"; return; fi
  "$PY" -m datagen.sample_positions "$pgn" --output "$out" --n-per-game "$NPG" --seed 1
}
export -f sample_one
export PY NPG

jobs_file="$(mktemp)"
i=0
for pgn in $(ls "$ALLOC"/train_shardrated_*.pgn.zst | sort); do
  printf '%s\t%s\n' "$pgn" "$(printf '%s/train-%03d.jsonl' "$OUT" "$i")" >> "$jobs_file"
  i=$((i + 1))
done
for pgn in $(ls "$ALLOC"/val_test_shardrated_*.pgn.zst | sort); do
  printf '%s\t%s\n' "$pgn" "$OUT/val_test-000.jsonl" >> "$jobs_file"
done

echo "[datagen] sampling $(wc -l < "$jobs_file") shards (16-way, n_per_game=$NPG)..."
xargs -P 16 -L 1 -a "$jobs_file" bash -c 'sample_one "$0" "$1"'
rm -f "$jobs_file"
echo "[datagen] train positions: $(cat "$OUT"/train-*.jsonl | wc -l)"
echo "[datagen] val_test positions: $(wc -l < "$OUT"/val_test-000.jsonl)"

# --- 2. build QA datasets --------------------------------------------------
"$PY" -m datagen.build_qa_dataset --config "$REPO/configs/datagen_stage1.yaml"

# --- 3. concatenate per-task train arrows into one (uniform 1/7) ------------
"$PY" - <<'PYEOF'
import glob, json, os
from datasets import load_from_disk, concatenate_datasets
root = "/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/data/stage1"
parts = sorted(glob.glob(f"{root}/training_datasets/*.arrow"))
dss = [load_from_disk(p) for p in parts]
print("[concat] per-task sizes:", {os.path.basename(p): len(d) for p, d in zip(parts, dss)})
combined = concatenate_datasets(dss)
out = f"{root}/train_all.arrow"
combined.save_to_disk(out)
# trainer reads only `pov` from the train dir's dataset_config.json
pov = json.load(open(f"{parts[0]}/dataset_config.json"))["pov"]
json.dump({"pov": pov, "split": "train", "note": "concatenated 7 per-task stage1 arrows",
           "records": len(combined), "tasks": [os.path.basename(p).replace('.arrow','') for p in parts]},
          open(f"{out}/dataset_config.json", "w"), indent=2)
print(f"[concat] -> {out}  ({len(combined):,} records, pov={pov})")
PYEOF
echo "[datagen] done -> data/stage1/ (train_all.arrow, val_dataset.arrow, test_dataset.arrow)"
