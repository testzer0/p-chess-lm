"""Create and save HuggingFace datasets for ChessLM SFT training.

Modes
-----
train   Generate a mixed training dataset from positions.jsonl.
eval    Generate an exhaustive eval dataset (every square + every piece type per position).
all     Both of the above.

Encoding variants
-----------------
Default (--pov False, --new-tok-in-query True):
    v2.1 — board-absolute SQUARE_TOKENS, special token embedded in question text.

--pov:
    v3 — POV-relative POV_SQUARE_TOKENS, no hidden-state flip at encode time.
         Special token always in query (implied by pov).

--no-new-tok-in-query (with --pov False):
    v2 legacy — board-absolute, plain-text square names in question.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset

from chesslm.utils.utils import ANSWER_SPECIAL_TOKENS
from chesslm.utils.generate_sft_data import generate_eval_set, sample_question_from_position

POSITIONS_PATH = Path(__file__).parent.parent / "raw_data" / "positions_v2.jsonl"
DATASETS_PATH  = Path(__file__).parent.parent / "datasets"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Create SFT datasets for ChessLM.")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--input", default=str(POSITIONS_PATH),
                        help="Path to positions.jsonl")
    parser.add_argument("--output-dir", default=str(DATASETS_PATH))
    parser.add_argument("--n-positions", type=int, default=250_000,
                        help="Total positions to process; total examples = n_positions × questions_per_position")
    parser.add_argument("--static-square-frac", type=float, default=0.75,
                        help="Fraction of positions allocated to static_square questions")
    parser.add_argument("--n-eval-positions", type=int, default=100,
                        help="Number of positions for the eval set")
    parser.add_argument("--questions-per-position", type=int, default=2,
                        help="QA samples generated per position")
    parser.add_argument("--seed", type=int, default=42)
    # Encoding variant flags
    parser.add_argument("--new-tok-in-query", action=argparse.BooleanOptionalAction, default=True,
                        help="Embed special tokens directly in the question text (default: on)")
    parser.add_argument("--pov", action="store_true", default=False,
                        help="Use POV-relative square tokens (v3); implies tok-in-query")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sq_question_type(pov: bool) -> str:
    return "static_square_pov" if pov else "static_square"

def _pc_question_type(pov: bool) -> str:
    return "static_piece_pov" if pov else "static_piece"


# ---------------------------------------------------------------------------
# Train dataset generation
# ---------------------------------------------------------------------------

def _generate_type(
    input_path: str, question_type: str, n_positions: int, seed: int,
    questions_per_position: int = 2,
    new_tok_in_query: bool = True,
    exclude: set = None,
) -> list[dict]:
    """Stream positions and generate QA pairs for exactly n_positions unique positions."""
    rng         = random.Random(seed)
    frequency   = defaultdict(int)
    records     = []
    seen        = set(exclude) if exclude else set()
    n_processed = 0

    with open(input_path) as f:
        for line in f:
            if n_processed >= n_positions:
                break
            start_fen, moves, end_fen = json.loads(line)
            key = (start_fen, tuple(moves))
            if key in seen:
                continue
            seen.add(key)
            n_processed += 1
            for _ in range(questions_per_position):
                sample = sample_question_from_position(
                    fen=end_fen,
                    question_type=question_type,
                    frequency=frequency,
                    rng=rng,
                    new_tok_in_query=new_tok_in_query,
                )
                for cls in sample["answer_class"]:
                    frequency[cls] += 1
                sample["start_fen"] = start_fen
                sample["moves"]     = moves
                records.append(sample)

    if n_processed < n_positions:
        print(f"  Warning: only {n_processed} positions available for {question_type} (requested {n_positions})")
    return records


def generate_train_dataset(args, exclude: set = None) -> Dataset:
    n_sq_pos = int(args.n_positions * args.static_square_frac)
    n_pc_pos = args.n_positions - n_sq_pos
    qpp      = args.questions_per_position
    ntiq     = args.new_tok_in_query or args.pov
    sq_qt    = _sq_question_type(args.pov)
    pc_qt    = _pc_question_type(args.pov)
    print(f"Generating train: {n_sq_pos} {sq_qt} × {qpp} = {n_sq_pos*qpp}  |  "
          f"{n_pc_pos} {pc_qt} × {qpp} = {n_pc_pos*qpp}  |  "
          f"{args.n_positions*qpp} total  (new_tok_in_query={ntiq})")

    sq_records = _generate_type(args.input, sq_qt, n_sq_pos, seed=args.seed,
                                questions_per_position=qpp, new_tok_in_query=ntiq, exclude=exclude)
    pc_records = _generate_type(args.input, pc_qt, n_pc_pos, seed=args.seed + 1,
                                questions_per_position=qpp, new_tok_in_query=ntiq, exclude=exclude)

    all_records = sq_records + pc_records
    random.Random(args.seed).shuffle(all_records)

    dataset = Dataset.from_list(all_records)
    out = Path(args.output_dir) / "train"
    dataset.save_to_disk(str(out))
    print(f"  Saved {len(dataset)} examples → {out}")
    return dataset


# ---------------------------------------------------------------------------
# Eval dataset generation
# ---------------------------------------------------------------------------

def generate_eval_dataset(args) -> tuple[Dataset, set]:
    """Returns (dataset, eval_trajectories) so train can exclude eval positions."""
    positions = []
    eval_keys = set()
    with open(args.input) as f:
        for line in f:
            entry = json.loads(line)
            key   = (entry[0], tuple(entry[1]))
            if key in eval_keys:
                continue
            eval_keys.add(key)
            positions.append(entry)
            if len(positions) == args.n_eval_positions:
                break

    sq_qt = _sq_question_type(args.pov)
    pc_qt = _pc_question_type(args.pov)
    ntiq  = args.new_tok_in_query or args.pov
    print(f"Generating eval: {len(positions)} positions × 76 questions = {len(positions)*76} total "
          f"({sq_qt}, {pc_qt}, new_tok_in_query={ntiq})")

    records = []
    for start_fen, moves, end_fen in positions:
        for qt in [sq_qt, pc_qt]:
            records.extend(generate_eval_set(
                fen=end_fen,
                question_type=qt,
                start_fen=start_fen,
                moves=moves,
                seed=args.seed,
                new_tok_in_query=ntiq,
            ))

    dataset = Dataset.from_list(records)
    out = Path(args.output_dir) / "eval"
    dataset.save_to_disk(str(out))
    print(f"  Saved {len(dataset)} examples → {out}")
    return dataset, eval_keys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print(f"Output directory: {args.output_dir}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write dataset config so training code knows which encode_positions mode to use
    config = {"pov": args.pov, "new_tok_in_query": args.new_tok_in_query or args.pov}
    with open(out_dir / "dataset_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Dataset config: {config}")

    eval_keys = None
    if args.mode in ("eval", "all"):
        _, eval_keys = generate_eval_dataset(args)

    if args.mode in ("train", "all"):
        generate_train_dataset(args, exclude=eval_keys)

    print(f"\nSpecial tokens in tokenizer: {len(ANSWER_SPECIAL_TOKENS)}")


if __name__ == "__main__":
    main()
