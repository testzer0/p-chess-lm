"""YAML-driven multi-split builder for Stage-1 QA datasets.

Output layout:

    config.output_path/
    ├── val_dataset.arrow/        # HF arrow shards + dataset_config.json
    ├── test_dataset.arrow/
    └── training_datasets/
        ├── piece_on_square.arrow/
        └── ...

Each `.arrow/` carries its own `dataset_config.json` (pov + slice + stats),
so the trainer reads `pov` directly without a tree walk.

Generation order: test → val → train. Test/val FENs join a `seen` set;
train skips those FENs but otherwise reuses positions freely. Train shards
cycle if the requested budget exceeds available positions.

Usage:

    python -m datagen.build_qa_dataset \\
        --config configs/datagen/stage1.yaml \\
        [--seed N] [--pov | --no-pov] [--output-path PATH] \\
        [--override KEY.PATH=VALUE ...]
"""
import argparse
import json
import random
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

import chess
import yaml
from datasets import Dataset
from tqdm import tqdm

from datagen.tasks import TASKS
from utils.board_representation import BoardRepr


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _coerce(s: str):
    """Coerce a string to int → float → bool → str (in that order)."""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _apply_override(cfg: dict, token: str) -> None:
    """Walk a dot-path into `cfg` and assign the (type-coerced) value."""
    key, sep, raw = token.partition("=")
    if not sep:
        raise ValueError(f"--override token missing '=': {token!r}")
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = _coerce(raw)


def _resolve_config(args) -> dict:
    """Load YAML, apply overrides, then layer in explicit CLI scalars."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    for token in args.override or []:
        _apply_override(cfg, token)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.pov is not None:
        cfg["pov"] = args.pov
    if args.output_path is not None:
        cfg["output_path"] = args.output_path

    return cfg


def _validate_config(cfg: dict) -> None:
    for key in ("seed", "pov", "output_path", "tasks", "splits"):
        if key not in cfg:
            raise ValueError(f"config missing required field: {key!r}")
    for task in cfg["tasks"]:
        if task not in TASKS:
            raise ValueError(f"unknown task in config: {task!r}; known: {sorted(TASKS)}")
        spec = cfg["tasks"][task]
        for k in ("num_positions", "num_questions"):
            if k not in spec:
                raise ValueError(f"tasks.{task} missing required field: {k!r}")
    for split in ("train", "val", "test"):
        if split not in cfg["splits"]:
            raise ValueError(f"splits missing required entry: {split!r}")
        if "shards" not in cfg["splits"][split]:
            raise ValueError(f"splits.{split} missing required field: 'shards'")
    for split in ("val", "test"):
        s = cfg["splits"][split]
        has_n = "num_positions" in s
        has_pt = "per_task" in s
        if has_n == has_pt:
            raise ValueError(
                f"splits.{split} must specify exactly one of "
                f"'num_positions' or 'per_task' (got num_positions={has_n}, per_task={has_pt})"
            )
        if has_pt and "num_questions" not in s:
            raise ValueError(f"splits.{split} with 'per_task' must also specify 'num_questions'")


def _resolve_shards(patterns) -> list[str]:
    """Expand a list of file paths / globs into a sorted unique list of paths."""
    if isinstance(patterns, str):
        patterns = [patterns]
    paths: list[str] = []
    for pattern in patterns:
        matched = sorted(glob(str(pattern)))
        if not matched:
            raise ValueError(f"no files matched shard pattern: {pattern!r}")
        paths.extend(matched)
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Shard streaming
# ---------------------------------------------------------------------------

def _load_positions(paths):
    """Stream (start_fen, moves, end_fen) triples from one or more JSONL files."""
    for path in paths:
        with open(path) as f:
            for line in f:
                start_fen, moves, end_fen = json.loads(line)
                yield start_fen, moves, end_fen


def _stream_filtered(paths, seen_fens):
    """Yield (start_fen, moves, end_fen) from paths, skipping triples whose
    end_fen is in seen_fens. seen_fens is read-only here."""
    for triple in _load_positions(paths):
        if triple[2] in seen_fens:
            continue
        yield triple


def _stream_cycling(paths, seen_fens, cycle_counter: list[int]):
    """Infinite version of _stream_filtered. cycle_counter[0] is incremented
    every time the shards are restarted from the beginning."""
    while True:
        cycle_counter[0] += 1
        any_yielded = False
        for triple in _stream_filtered(paths, seen_fens):
            any_yielded = True
            yield triple
        if not any_yielded:
            raise RuntimeError(
                "All shard positions are filtered out by the seen-FEN set; "
                "cannot make progress."
            )


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def _build_history(start_fen: str, moves: list[str]) -> list[str]:
    if not moves:
        return []
    board = chess.Board(start_fen)
    out = [board.fen()]
    for uci in moves[:-1]:
        board.push_uci(uci)
        out.append(board.fen())
    return out


def _record(s: dict, end_fen: str, history: list[str]) -> dict:
    return {
        "fen":      end_fen,
        "history":  history,
        "prompt":   s["question"],
        "response": s["answer"],
        "extra": {
            "task":         s["question_type"],
            "answer_class": s["answer_class"],
        },
    }


def _bump_frequency(frequency: dict, s: dict) -> None:
    for cls in s["answer_class"]:
        frequency[cls] += 1


# ---------------------------------------------------------------------------
# Eval split builds (test / val)
# ---------------------------------------------------------------------------

def _build_eval_split(label: str, split_cfg: dict, task_names: list[str],
                      pov: bool, rng: random.Random, seen_fens: set) -> tuple[list[dict], set, dict]:
    """Build records for 'test' or 'val'. Returns (records, new_fens, summary)."""
    shards = _resolve_shards(split_cfg["shards"])
    records: list[dict] = []
    new_fens: set = set()
    per_task_counts: dict[str, int] = defaultdict(int)

    if "num_positions" in split_cfg:
        # Mode A: read N unique positions; for each, run sample_all per task.
        n = split_cfg["num_positions"]
        # One frequency dict per task (entity-level balance is per-task).
        freq = {t: defaultdict(int) for t in task_names}
        pbar = tqdm(total=n, desc=f"{label} (mode=num_positions)", unit="pos")
        for start_fen, moves, end_fen in _stream_filtered(shards, seen_fens | new_fens):
            if len(new_fens) >= n:
                break
            board = BoardRepr.from_fen(end_fen, pov=pov)
            history = _build_history(start_fen, moves)
            for task in task_names:
                module = TASKS[task]
                recs = module.sample_all(board, freq[task], rng)
                for s in recs:
                    _bump_frequency(freq[task], s)
                    records.append(_record(s, end_fen, history))
                per_task_counts[task] += len(recs)
            new_fens.add(end_fen)
            pbar.update(1)
        pbar.close()
        summary = {
            "mode":             "num_positions",
            "positions":        len(new_fens),
            "records":          len(records),
            "per_task_records": dict(per_task_counts),
        }
    else:
        # Mode B: each task pulls per_task positions and emits num_questions per pos.
        k = split_cfg["per_task"]
        q = split_cfg["num_questions"]
        # Positions are consumed sequentially across tasks via a single shared stream.
        # FEN dedup against `seen_fens ∪ already-collected new_fens` (cumulative).
        stream = _stream_filtered(shards, seen_fens)  # not cycling — eval should not cycle
        per_task_positions: dict[str, int] = {}
        for task in task_names:
            module = TASKS[task]
            max_q = module.MAX_UNIQUE_QUERIES
            n_q = min(q, max_q)
            if q > max_q:
                print(f"  WARN: {label}: task {task} num_questions={q} > MAX_UNIQUE_QUERIES={max_q}; capping to {max_q}")
            freq: dict = defaultdict(int)
            taken = 0
            pbar = tqdm(total=k, desc=f"{label} (mode=per_task, task={task})", unit="pos")
            while taken < k:
                try:
                    start_fen, moves, end_fen = next(stream)
                except StopIteration:
                    print(f"  WARN: {label}: task {task} exhausted shards after {taken}/{k} positions")
                    break
                if end_fen in new_fens:
                    continue
                board = BoardRepr.from_fen(end_fen, pov=pov)
                history = _build_history(start_fen, moves)
                recs = module.sample_n(board, freq, rng, n_q)
                for s in recs:
                    _bump_frequency(freq, s)
                    records.append(_record(s, end_fen, history))
                per_task_counts[task] += len(recs)
                new_fens.add(end_fen)
                taken += 1
                pbar.update(1)
            pbar.close()
            per_task_positions[task] = taken
        summary = {
            "mode":               "per_task",
            "per_task":           k,
            "num_questions":      q,
            "records":            len(records),
            "per_task_records":   dict(per_task_counts),
            "per_task_positions": per_task_positions,
        }
    return records, new_fens, summary


# ---------------------------------------------------------------------------
# Train split build
# ---------------------------------------------------------------------------

def _build_train(train_cfg: dict, tasks_cfg: dict, pov: bool,
                 rng: random.Random, seen_fens: set) -> tuple[dict[str, list[dict]], dict]:
    """Build per-task train datasets. Positions are consumed sequentially
    across tasks; shards cycle on exhaustion."""
    shards = _resolve_shards(train_cfg["shards"])
    cycle_counter = [0]
    stream = _stream_cycling(shards, seen_fens, cycle_counter)

    per_task_records: dict[str, list[dict]] = {t: [] for t in tasks_cfg}
    per_task_summary: dict[str, dict] = {}
    total_records = 0

    for task, spec in tasks_cfg.items():
        module = TASKS[task]
        num_positions = spec["num_positions"]
        num_questions = spec["num_questions"]
        max_q = module.MAX_UNIQUE_QUERIES
        n_q = min(num_questions, max_q)
        capped = num_questions > max_q
        if capped:
            print(f"  WARN: train: task {task} num_questions={num_questions} > MAX_UNIQUE_QUERIES={max_q}; capping to {max_q}")

        freq: dict = defaultdict(int)
        taken = 0
        pbar = tqdm(total=num_positions, desc=f"train (task={task})", unit="pos")
        while taken < num_positions:
            start_fen, moves, end_fen = next(stream)
            board = BoardRepr.from_fen(end_fen, pov=pov)
            history = _build_history(start_fen, moves)
            recs = module.sample_n(board, freq, rng, n_q)
            for s in recs:
                _bump_frequency(freq, s)
                per_task_records[task].append(_record(s, end_fen, history))
            taken += 1
            pbar.update(1)
        pbar.close()

        per_task_summary[task] = {
            "positions_consumed":   taken,
            "records":              len(per_task_records[task]),
            "capped_num_questions": capped,
        }
        total_records += len(per_task_records[task])

    return per_task_records, {
        "per_task":             per_task_summary,
        "total_records":        total_records,
        "train_shards_cycled":  max(0, cycle_counter[0] - 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_dataset(records: list[dict], path: Path) -> None:
    if records:
        Dataset.from_list(records).save_to_disk(str(path))
    else:
        Dataset.from_dict({"fen": [], "history": [], "prompt": [], "response": [], "extra": []}).save_to_disk(str(path))


def _save_arrow_config(arrow_dir: Path, payload: dict) -> None:
    """Write the per-arrow dataset_config.json. `payload` is already the full
    dict to serialize — each split/task assembles only the slice it needs
    (the trainer reads `pov`; debugging tooling gets context + stats)."""
    with open(arrow_dir / "dataset_config.json", "w") as f:
        json.dump(payload, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="YAML-driven Stage-1 QA dataset builder.")
    p.add_argument("--config",      required=True, type=Path)
    p.add_argument("--seed",        type=int,        default=None)
    p.add_argument("--pov",         dest="pov", action="store_true",  default=None)
    p.add_argument("--no-pov",      dest="pov", action="store_false", default=None)
    p.add_argument("--output-path", type=str,        default=None)
    p.add_argument("--override",    nargs="*",       default=None,
                   help="Dotted KEY.PATH=VALUE overrides for nested YAML fields.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = _resolve_config(args)
    _validate_config(cfg)

    pov         = bool(cfg["pov"])
    seed        = int(cfg["seed"])
    output_path = Path(cfg["output_path"])
    task_names  = list(cfg["tasks"].keys())
    tasks_cfg   = cfg["tasks"]

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "training_datasets").mkdir(parents=True, exist_ok=True)

    # Independent RNGs per split so re-running with a different val budget
    # doesn't shift the train sequence.
    rng_test  = random.Random(seed + 1)
    rng_val   = random.Random(seed + 2)
    rng_train = random.Random(seed + 3)

    seen_fens: set = set()
    build_started = time.time()

    print(f"[1/3] Building test split...")
    test_records, test_fens, test_summary = _build_eval_split(
        "test", cfg["splits"]["test"], task_names, pov, rng_test, seen_fens,
    )
    seen_fens |= test_fens
    test_dir = output_path / "test_dataset.arrow"
    _save_dataset(test_records, test_dir)
    _save_arrow_config(test_dir, {
        "pov":           pov,
        "seed":          seed,
        "split":         "test",
        "tasks":         task_names,
        "split_spec":    cfg["splits"]["test"],
        "summary_stats": test_summary,
    })
    print(f"      → {len(test_records)} records across {len(test_fens)} positions")

    print(f"[2/3] Building val split...")
    val_records, val_fens, val_summary = _build_eval_split(
        "val", cfg["splits"]["val"], task_names, pov, rng_val, seen_fens,
    )
    seen_fens |= val_fens
    val_dir = output_path / "val_dataset.arrow"
    _save_dataset(val_records, val_dir)
    _save_arrow_config(val_dir, {
        "pov":           pov,
        "seed":          seed,
        "split":         "val",
        "tasks":         task_names,
        "split_spec":    cfg["splits"]["val"],
        "summary_stats": val_summary,
    })
    print(f"      → {len(val_records)} records across {len(val_fens)} positions")

    print(f"[3/3] Building train split (per-task)...")
    train_records, train_summary = _build_train(
        cfg["splits"]["train"], tasks_cfg, pov, rng_train, seen_fens,
    )
    for task, recs in train_records.items():
        task_dir = output_path / "training_datasets" / f"{task}.arrow"
        _save_dataset(recs, task_dir)
        _save_arrow_config(task_dir, {
            "pov":                 pov,
            "seed":                seed,
            "split":               "train",
            "task":                task,
            "task_spec":           tasks_cfg[task],
            "train_shards":        cfg["splits"]["train"]["shards"],
            "summary_stats":       train_summary["per_task"][task],
            "train_shards_cycled": train_summary["train_shards_cycled"],
        })
    print(f"      → {train_summary['total_records']} records across "
          f"{sum(s['positions_consumed'] for s in train_summary['per_task'].values())} positions")

    build_finished = time.time()
    print(f"Done in {build_finished - build_started:.1f}s. Output: {output_path}")


if __name__ == "__main__":
    main()
