# `datagen/` — Stage-1 QA dataset pipeline

Two stages: raw positions → QA examples.

```
lichess PGN / PGN.zst
        │
        ▼  sample_positions.py
positions.jsonl   ([start_fen, moves, end_fen] per line)
        │
        ▼  build_qa_dataset.py   (YAML-driven)
<output_path>/
├── test_dataset.arrow/         HF dir + dataset_config.json
├── val_dataset.arrow/
└── training_datasets/
    ├── piece_on_square.arrow/
    ├── square_of_piece.arrow/
    ├── piece_on_file.arrow/
    ├── piece_on_rank.arrow/
    ├── piece_on_diagonal.arrow/
    ├── piece_count.arrow/
    └── material_count.arrow/
```

Per-task QA logic lives in [`tasks/`](tasks/README.md); shared prose
helpers + line-task fact builder live in `prose.py`.

---

## `sample_positions.py`

Streams a Lichess PGN file and writes random sampled positions to JSONL.
Each line is `[start_fen, move_list, end_fen]`:

- `start_fen` — FEN up to 7 plies before the sampled position
- `move_list` — UCI moves taking start_fen → end_fen
- `end_fen`   — the sampled position

Format matches what `lc0` expects (8 history slots = start board + ≤7
moves). Storing more than 7 moves is wasteful.

**Usage:**

```bash
# Sanity check: load one random game, show a random ply
python -m datagen.sample_positions data/lichess/games.pgn.zst --interactive

# Bulk: sample positions across all games in the file
python -m datagen.sample_positions data/lichess/games.pgn.zst \
    --output data/lichess/train-001.jsonl \
    --n-per-game 1 \
    --max-games 500000 \
    --seed 42
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `pgn` | — | Path to `.pgn` or `.pgn.zst` file (positional). |
| `--output` | `positions.jsonl` | Output JSONL path. |
| `--n-per-game` | `1` | Positions sampled per game. |
| `--max-games` | all | Stop after this many games (1000 in interactive). |
| `--seed` | `42` | RNG seed (random in interactive mode). |
| `--interactive` | off | Show one position + board; don't write JSONL. |

---

## `build_qa_dataset.py`

YAML-driven multi-split builder. One command produces test / val / train
datasets, with FEN-level dedup between splits.

**Usage:**

```bash
# Run the config verbatim
python -m datagen.build_qa_dataset --config configs/sample_datagen.yaml

# Reseed
python -m datagen.build_qa_dataset --config configs/sample_datagen.yaml --seed 7

# Tweak any nested field from the CLI without editing the YAML:
python -m datagen.build_qa_dataset --config configs/sample_datagen.yaml \
    --override tasks.piece_on_square.num_positions=50000 \
               splits.val.num_positions=2000 \
               output_path=data/stage1_seed7
```

Flat scalar flags (`--seed`, `--pov` / `--no-pov`, `--output-path`) carry
sentinel defaults — anything not passed falls back to the YAML value.

### Config format

A complete reference config lives at
[`configs/sample_datagen.yaml`](../configs/sample_datagen.yaml). Required top-level keys:

| Key | Type | Notes |
|---|---|---|
| `seed`        | int  | Reproducibility. |
| `pov`         | bool | `true` → POV-relative tokens; `false` → board-absolute. |
| `output_path` | str  | Root directory for the generated datasets. |
| `tasks`       | dict | `{task_name: {num_positions, num_questions}}` for the train split. |
| `splits`      | dict | `{train, val, test}` — each names its shards + (for val/test) sampling mode. |

`tasks` only knows about the task names registered in
[`tasks/__init__.py`](tasks/__init__.py:TASKS).

### Eval modes (val + test)

Each of `val` and `test` picks exactly one mode:

- **Mode A — `num_positions: N`**: load `N` unique FENs (skipping any
  already in `seen`); for each FEN, every task emits `sample_all(board)`
  records (every distinct entity). Total records ≈
  `N × Σ MAX_UNIQUE_QUERIES`. Good for exhaustive eval sets where each
  position is fully covered.

- **Mode B — `per_task: N` + `num_questions: K`**: each task pulls `N`
  positions sequentially from the shared stream; each position emits `K`
  distinct queries (via `sample_n`). `K` is silently capped at the task's
  `MAX_UNIQUE_QUERIES` (one warning per cap). Good for cheaper eval where
  not every entity needs coverage.

### Dedup invariants

- Test runs first → its FENs join `seen_fens`.
- Val runs next, skipping FENs in `seen_fens`; its FENs join `seen_fens` too.
- Train runs last, skipping FENs in `seen_fens`. Within train,
  FENs may repeat freely (no intra-train dedup; that set would grow
  unboundedly).
- Train shards **cycle** when exhausted (the build never stops early
  because shards ran out — it cycles back to the first shard and
  continues, skipping FENs in `seen_fens`).

### Per-arrow `dataset_config.json`

Each `.arrow/` directory carries its own self-describing config. The
trainer only ever needs `pov`, which is at the top level:

```json
{
  "pov":        true,
  "seed":       0,
  "split":      "train",
  "task":       "piece_on_square",
  "task_spec":  { "num_positions": 200000, "num_questions": 4 },
  "summary_stats": {
    "positions_consumed":   200000,
    "records":              800000,
    "capped_num_questions": false
  },
  "train_shards_cycled": 0
}
```

The full input YAML for repro lives on disk at the user's `--config` path
— not duplicated into every arrow.
