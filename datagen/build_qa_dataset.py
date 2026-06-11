"""Build one (question_type, encoding) Arrow dataset from a positions shard.

Task-agnostic driver. Per-task logic (entity enumeration, frequency-aware
weighting, prompt/response prose) lives in `datagen/tasks/*.py`; this file
only wires together TASKS[args.question_type] over the positions stream and
serializes records to the project-wide dataset spec:

    fen      : str        — current/final position the question is about
    history  : list[str]  — prior board FENs, oldest->newest, excluding fen
                            (empty list = static position)
    prompt   : str        — question text
    response : str        — gold answer (no trailing period; EOS appended at
                            tokenization)
    extra    : dict       — eval metadata (question_type, answer_class);
                            NOT fed to the model.

Output: HF Arrow dataset at {output_dir}/{question_type}/{abs|rel}/train/
plus a sibling dataset_config.json.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import chess
from datasets import Dataset

from datagen.tasks import TASKS
from utils.board_representation import BoardRepr


def _load_positions(path: str):
    with open(path) as f:
        for line in f:
            start_fen, moves, end_fen = json.loads(line)
            yield start_fen, moves, end_fen


def _history(start_fen: str, moves: list) -> list:
    """All board FENs from start through penultimate ply (final excluded).

    Maps the (start_fen, moves) shape from sample_positions.py to the
    spec's `history` list. moves=[] -> [] (static position).
    """
    if not moves:
        return []
    board = chess.Board(start_fen)
    out = [board.fen()]
    for uci in moves[:-1]:
        board.push_uci(uci)
        out.append(board.fen())
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Build one (question_type, encoding) Arrow dataset.")
    p.add_argument("--positions", required=True,
                   help="Path to positions JSONL (one [start_fen, moves, end_fen] per line)")
    p.add_argument("--question-type", required=True, choices=list(TASKS),
                   help="Question type to generate")
    p.add_argument("--absolute", action="store_true",
                   help="Use absolute square/piece tokens (default: POV)")
    p.add_argument("--questions-per-position", type=int, default=2)
    p.add_argument("--n-positions", type=int, default=-1,
                   help="Cap on positions to consume (default: all)")
    p.add_argument("--output-dir", required=True,
                   help="Root; dataset is written to {output_dir}/{question_type}/{abs|rel}/train/")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = random.Random(args.seed)
    pov  = not args.absolute

    sample_one = TASKS[args.question_type]
    encoding   = "rel" if pov else "abs"
    out_root   = Path(args.output_dir) / args.question_type / encoding
    out_root.mkdir(parents=True, exist_ok=True)

    positions = _load_positions(args.positions)
    if args.n_positions > 0:
        positions = (p for i, p in enumerate(positions) if i < args.n_positions)

    print(f"Generating {args.question_type} / {encoding}  "
          f"(pov={pov}, qpp={args.questions_per_position})")

    frequency: dict = defaultdict(int)
    seen: set = set()
    records: list = []

    for start_fen, moves, end_fen in positions:
        key = (start_fen, tuple(moves))
        if key in seen:
            continue
        seen.add(key)

        board   = BoardRepr.from_fen(end_fen, pov)
        history = _history(start_fen, moves)
        for _ in range(args.questions_per_position):
            s = sample_one(board, frequency, rng)
            for cls in s["answer_class"]:
                frequency[cls] += 1
            records.append({
                "fen":      end_fen,
                "history":  history,
                "prompt":   s["question"],
                "response": s["answer"],
                "extra": {
                    "task":         s["question_type"],
                    "answer_class": s["answer_class"],
                },
            })

    random.Random(args.seed).shuffle(records)

    dataset   = Dataset.from_list(records)
    train_dir = out_root / "train"
    dataset.save_to_disk(str(train_dir))
    print(f"  saved {len(dataset)} examples -> {train_dir}")

    cfg = {
        "pov":                    pov,
        "encoding":               encoding,
        "question_type":          args.question_type,
        "questions_per_position": args.questions_per_position,
        "n_examples":             len(dataset),
        "seed":                   args.seed,
    }
    with open(out_root / "dataset_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  wrote config -> {out_root / 'dataset_config.json'}")


if __name__ == "__main__":
    main()
