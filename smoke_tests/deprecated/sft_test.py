"""Smoke tests for generate_sft_data.py.

Checks:
  1. _generate_position_dict  — square_to_piece and piece_to_squares are consistent
  2. static_square            — output format, parse tag, answer_class
  3. static_piece             — output format, parse tag, answer_class (present + absent)
  4. frequency balancing      — after many samples, answer_class counts converge
  5. end-to-end main()        — runs on 100 positions and writes valid JSONL
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import chess

from chesslm.utils.utils import (
    SQUARE_TOKENS, PIECE_TOKENS, EMPTY_TOKEN, ANSWER_SPECIAL_TOKENS,
    _generate_position_dict,
)
from chesslm.utils.generate_sft_data import sample_question_from_position

POSITIONS_PATH = Path(__file__).parent.parent / "raw_data" / "positions.jsonl"

_STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def _load_positions(n=500):
    positions = []
    with open(POSITIONS_PATH) as f:
        for line in f:
            positions.append(json.loads(line))
            if len(positions) == n:
                break
    return positions

def _sparse_fen(positions):
    """Return the end_fen with the fewest pieces (most absent piece types)."""
    return min(positions, key=lambda p: chess.Board(p[2]).occupied.bit_count())[2]


# ---------------------------------------------------------------------------
# Test 1: _generate_position_dict
# ---------------------------------------------------------------------------

def test_position_dict():
    print("=== test_position_dict ===")

    sq2p, p2sq = _generate_position_dict(_STARTING_FEN)

    assert len(sq2p) == 64, f"Expected 64 squares, got {len(sq2p)}"

    # Every value is a known token
    valid = set(PIECE_TOKENS) | {EMPTY_TOKEN}
    for sq, tok in sq2p.items():
        assert tok in valid, f"Unknown token {tok!r} for square {sq}"

    # piece_to_squares is the inverse of square_to_piece
    for tok, squares in p2sq.items():
        for sq in squares:
            assert sq2p[sq] == tok, f"Inconsistency: {sq} → {sq2p[sq]}, expected {tok}"

    # Starting position: 16 pawns, 4 rooks, etc.
    assert len(p2sq["<PIECE_WP>"]) == 8
    assert len(p2sq["<PIECE_BP>"]) == 8
    assert len(p2sq["<PIECE_WK>"]) == 1
    assert len(p2sq["<PIECE_BK>"]) == 1

    # 32 pieces + 32 empty
    n_empty = sum(1 for t in sq2p.values() if t == EMPTY_TOKEN)
    assert n_empty == 32, f"Expected 32 empty squares, got {n_empty}"

    print("  square_to_piece and piece_to_squares consistent  OK")


# ---------------------------------------------------------------------------
# Test 2: static_square output format
# ---------------------------------------------------------------------------

def test_static_square():
    print("=== test_static_square ===")
    rng = random.Random(0)
    positions = _load_positions(10)

    for fen in [_STARTING_FEN] + [p[2] for p in positions[:3]]:
        for _ in range(20):
            s = sample_question_from_position(fen, "static_square", rng=rng)

            assert "question" in s and "answer" in s
            assert s["question_type"] == "static_square"

            # answer ends with "\n\n<SQUARE_XY><PIECE_or_EMPTY>"
            lines = s["answer"].split("\n\n")
            assert len(lines) == 2, f"Expected 2 parts split by \\n\\n, got: {s['answer']!r}"
            parse_tag = lines[1]

            # parse_tag = <SQUARE_XY><PIECE_or_EMPTY>
            sq_tok    = next((t for t in SQUARE_TOKENS if parse_tag.startswith(t)), None)
            assert sq_tok is not None, f"No square token at start of parse tag: {parse_tag!r}"
            piece_tok = parse_tag[len(sq_tok):]
            assert piece_tok in set(PIECE_TOKENS) | {EMPTY_TOKEN}, \
                f"Unexpected piece token: {piece_tok!r}"

            # answer_class is a list with one element matching the piece token
            assert s["answer_class"] == [piece_tok], \
                f"answer_class {s['answer_class']} != [{piece_tok}]"

    print("  format, parse tag, answer_class all correct  OK")


# ---------------------------------------------------------------------------
# Test 3: static_piece output format (present and absent)
# ---------------------------------------------------------------------------

def test_static_piece():
    print("=== test_static_piece ===")
    rng = random.Random(0)
    sparse_fen = _sparse_fen(_load_positions(200))

    seen_present = seen_absent = False

    for _ in range(100):
        s = sample_question_from_position(sparse_fen, "static_piece", rng=rng)

        assert s["question_type"] == "static_piece"
        lines = s["answer"].split("\n\n")
        assert len(lines) == 2, f"Bad answer format: {s['answer']!r}"
        parse_tag = lines[1]

        # parse_tag = <PIECE_XY> followed by one or more <SQUARE_XY> or <EMPTY>
        piece_tok = next((t for t in PIECE_TOKENS if parse_tag.startswith(t)), None)
        assert piece_tok is not None, f"No piece token at start: {parse_tag!r}"
        rest = parse_tag[len(piece_tok):]

        if rest == EMPTY_TOKEN:
            assert s["answer_class"] == [EMPTY_TOKEN], \
                f"absent answer_class should be [EMPTY_TOKEN], got {s['answer_class']}"
            seen_absent = True
        else:
            # rest should be one or more square tokens
            remaining = rest
            sq_list = []
            while remaining:
                sq_tok = next((t for t in SQUARE_TOKENS if remaining.startswith(t)), None)
                assert sq_tok is not None, f"Unexpected content in parse tag: {remaining!r}"
                sq_list.append(sq_tok)
                remaining = remaining[len(sq_tok):]
            assert s["answer_class"] == sq_list, \
                f"answer_class {s['answer_class']} != {sq_list}"
            seen_present = True

    assert seen_present, "Never saw a present-piece answer"
    assert seen_absent,  "Never saw an absent-piece answer"
    print("  present and absent branches correct  OK")


# ---------------------------------------------------------------------------
# Test 4: frequency balancing for static_square
# ---------------------------------------------------------------------------

def test_frequency_balancing():
    print("=== test_frequency_balancing ===")

    positions = _load_positions(5000)
    rng = random.Random(42)

    for qt in ["static_square", "static_piece"]:
        frequency = defaultdict(int)
        counts    = defaultdict(int)

        for _, _, end_fen in positions:
            s = sample_question_from_position(end_fen, qt, frequency=frequency, rng=rng)
            for cls in s["answer_class"]:
                frequency[cls] += 1
                counts[cls]    += 1

        total   = sum(counts.values())
        n_cls   = len(counts)
        ideal   = total / n_cls
        max_dev = max(abs(c - ideal) / ideal for c in counts.values())

        print(f"\n  [{qt}]  {n_cls} classes, {total} samples, ideal={ideal:.1f}")
        for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "#" * int(40 * cnt / max(counts.values()))
            print(f"    {cls:20s}  {cnt:5d}  {bar}")
        print(f"  max relative deviation from uniform: {max_dev:.2%}")

        assert n_cls >= 10, f"Too few classes seen: {n_cls}"
        if qt == "static_square":
            assert max_dev < 0.5, f"Balancing too skewed: {max_dev:.2%}"

    print("\n  frequency balancing working  OK")


# ---------------------------------------------------------------------------
# Test 5: end-to-end main()
# ---------------------------------------------------------------------------

def test_main():
    print("=== test_main ===")
    import subprocess

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        out_path = f.name

    result = subprocess.run(
        [sys.executable, "-m", "chesslm.utils.generate_sft_data",
         "--input", str(POSITIONS_PATH),
         "--output", out_path,
         "--n-positions", "50",
         "--questions-per-position", "2"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    assert result.returncode == 0, f"main() failed:\n{result.stderr}"

    lines = Path(out_path).read_text().strip().splitlines()
    assert len(lines) == 100, f"Expected 100 lines, got {len(lines)}"

    required_keys = {"question", "answer", "question_type", "answer_class", "fen", "start_fen", "moves"}
    for line in lines:
        row = json.loads(line)
        missing = required_keys - row.keys()
        assert not missing, f"Missing keys: {missing}"
        assert isinstance(row["answer_class"], list)

    Path(out_path).unlink()
    print(f"  wrote and validated 100 QA pairs  OK")


# ---------------------------------------------------------------------------
# Test 6: create_sft_dataset.py (train + eval)
# ---------------------------------------------------------------------------

def test_create_dataset():
    print("=== test_create_dataset ===")
    import subprocess
    from datasets import Dataset

    with tempfile.TemporaryDirectory() as out_dir:
        result = subprocess.run(
            [sys.executable, "-m", "chesslm.utils.create_sft_dataset",
             "--mode", "all",
             "--input", str(POSITIONS_PATH),
             "--output-dir", out_dir,
             "--n-train", "40",
             "--static-square-frac", "0.75",
             "--n-eval-positions", "5",
             "--questions-per-position", "2"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert result.returncode == 0, f"create_sft_dataset failed:\n{result.stderr}"

        # Train: 40 total (30 static_square + 10 static_piece), shuffled
        train = Dataset.load_from_disk(str(Path(out_dir) / "train"))
        assert len(train) == 40, f"Expected 40 train examples, got {len(train)}"
        types = set(train["question_type"])
        assert types == {"static_square", "static_piece"}, f"Unexpected types: {types}"
        print(f"  train: {len(train)} examples, types={types}  OK")

        # Eval: 5 positions × 76 questions = 380
        eval_ds = Dataset.load_from_disk(str(Path(out_dir) / "eval"))
        assert len(eval_ds) == 5 * 76, f"Expected {5*76} eval examples, got {len(eval_ds)}"
        required_keys = {"question", "answer", "question_type", "answer_class", "fen", "start_fen", "moves"}
        assert required_keys <= set(eval_ds.column_names), f"Missing keys: {required_keys - set(eval_ds.column_names)}"
        print(f"  eval: {len(eval_ds)} examples  OK")

        # Holdout: eval trajectories must never appear in train
        eval_keys  = {(r["start_fen"], tuple(r["moves"])) for r in eval_ds}
        train_keys = {(r["start_fen"], tuple(r["moves"])) for r in train}
        overlap = eval_keys & train_keys
        assert not overlap, f"Eval/train trajectory overlap: {overlap}"
        print(f"  eval trajectories are held out from train  OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    test_position_dict()
    print()
    test_static_square()
    print()
    test_static_piece()
    print()
    test_frequency_balancing()
    print()
    test_main()
    print()
    test_create_dataset()
    print()
    print("All SFT smoke tests passed!")


if __name__ == "__main__":
    main()
