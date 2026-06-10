"""Distribution audit for build_qa_dataset's inverse-frequency weighter.

For every registered question_type, generates a few thousand training examples
on a real positions shard and audits the empirical distribution over that
question_type's ANSWER classes.

What counts as an "answer class"
---------------------------------
Only tokens the weighter actually tries to balance. Query-side tokens (the
queried square in piece_on_square, the queried piece in square_of_piece, the
start/end tokens that identify a file/rank/diagonal) are NOT balanced by the
weighter — they reflect natural board geometry — and are deliberately excluded
from this audit. <EMPTY> IS an answer class everywhere (open files, absent
pieces, empty squares).

Per question_type:
  piece_on_square         13 classes — 12 piece tokens + <EMPTY>
  square_of_piece         65 classes — 64 square tokens + <EMPTY>
  pieces_on_*_direct/cot  13 classes — 12 piece tokens + <EMPTY>

Threshold model
---------------
Per-class sampling variance grows as 1/sqrt(N/C); to absorb this we scale the
user's --base-ratio (default 3.0) by sqrt(C/13). The 13 baseline is the
piece-kind class count where 3.0 is empirically loose for ~3k samples:
  13 classes → threshold = base
  65 classes → threshold = base × sqrt(65/13) ≈ 2.24 × base

WARN fires for a (qt, encoding) when either:
  - any expected class is entirely missing from the empirical distribution
  - max/min count ratio over present classes exceeds the scaled threshold

The script prints full per-QT tables regardless. Default exit code is 0;
pass --strict to exit 1 when any (qt, encoding) warns.

Usage
-----
python chesslm/smoke_tests/test_qa_distribution.py
python chesslm/smoke_tests/test_qa_distribution.py --strict
python chesslm/smoke_tests/test_qa_distribution.py --n-positions 2000 --pov both
python chesslm/smoke_tests/test_qa_distribution.py --question-types piece_on_square square_of_piece
"""
import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chesslm.utils.build_qa_dataset import (
    QUESTION_REGISTRY,
    _load_positions,
    generate_train_records,
)
from chesslm.utils.utils import (
    EMPTY_TOKEN,
    PIECE_TOKENS,
    POV_SQUARE_TOKENS,
    SQUARE_TOKENS,
)


# ---------------------------------------------------------------------------
# Per-QT answer-class extraction
# ---------------------------------------------------------------------------

def _expected_answer_classes(qt: str, pov: bool) -> set[str]:
    """The closed set of tokens the weighter tries to balance for this QT."""
    if qt == "piece_on_square":
        return set(PIECE_TOKENS) | {EMPTY_TOKEN}
    if qt == "square_of_piece":
        squares = POV_SQUARE_TOKENS if pov else SQUARE_TOKENS
        return set(squares) | {EMPTY_TOKEN}
    if qt.startswith("pieces_on_"):
        return set(PIECE_TOKENS) | {EMPTY_TOKEN}
    raise ValueError(f"unknown question_type: {qt!r}")


def _answer_tokens_from_record(rec: dict) -> list[str]:
    """Slice answer_class to drop query-side tokens.

    The slicing convention mirrors how each build_qa_fn assembles answer_class:
      piece_on_square : [sq_tok,          piece_or_empty]  → drop 1st
      square_of_piece : [piece_tok,       sq_tok…|empty]   → drop 1st
      pieces_on_*     : [start, end,      pieces…|empty]   → drop first 2
    """
    qt = rec["question_type"]
    ac = rec["answer_class"]
    if qt == "piece_on_square":    return ac[1:]
    if qt == "square_of_piece":    return ac[1:]
    if qt.startswith("pieces_on_"): return ac[2:]
    raise ValueError(f"unknown question_type: {qt!r}")


def _scaled_threshold(base: float, n_classes: int) -> float:
    return base * max(1.0, (n_classes / 13.0) ** 0.5)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure(qt: str, pov: bool, positions_path: str,
            n_positions: int, qpp: int, seed: int) -> tuple[Counter, int, int]:
    rng       = random.Random(seed)
    positions = list(_load_positions(positions_path))[:n_positions]
    records   = generate_train_records(positions, qt, pov, qpp, rng)

    expected = _expected_answer_classes(qt, pov)
    counts: Counter = Counter()
    n_unexpected = 0
    for rec in records:
        for tok in _answer_tokens_from_record(rec):
            if tok in expected:
                counts[tok] += 1
            else:
                n_unexpected += 1
    return counts, len(records), n_unexpected


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_distribution(qt: str, pov: bool, counts: Counter,
                        expected: set[str], n_records: int,
                        n_unexpected: int, base_ratio: float, head: int) -> dict:
    enc          = "rel" if pov else "abs"
    n_classes    = len(expected)
    threshold    = _scaled_threshold(base_ratio, n_classes)
    present      = set(counts.keys())
    missing      = expected - present
    items        = sorted(counts.items(), key=lambda kv: -kv[1])
    total_tokens = sum(counts.values())

    if items:
        hi_tok, hi_n = items[0]
        lo_tok, lo_n = items[-1]
        ratio = hi_n / lo_n if lo_n > 0 else float("inf")
    else:
        hi_tok = lo_tok = None
        hi_n = lo_n = 0
        ratio = float("inf")

    warn = bool(missing) or ratio > threshold

    print(f"\n=== {qt} / {enc} ===")
    print(f"  records={n_records:6d}  answer_tokens={total_tokens:6d}  "
          f"classes={len(present)}/{n_classes}  missing={len(missing)}")
    print(f"  ratio={ratio:7.2f}   threshold={threshold:6.2f}   "
          f"{'WARN' if warn else 'OK'}")
    if n_unexpected:
        print(f"  [!] {n_unexpected} unexpected tokens in answer_class (not in expected set)")

    if items:
        max_show = 2 * head
        if n_classes <= max_show:
            print("  counts (all):")
            for tok, cnt in items:
                print(f"    {tok:<16} {cnt}")
        else:
            print(f"  top {head}:")
            for tok, cnt in items[:head]:
                print(f"    {tok:<16} {cnt}")
            print(f"  bot {head}:")
            for tok, cnt in items[-head:]:
                print(f"    {tok:<16} {cnt}")

    if missing:
        sample = sorted(missing)[:8]
        more   = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        print(f"  missing: {sample}{more}")

    return {
        "qt": qt, "enc": enc, "n_records": n_records,
        "n_classes": n_classes, "present": len(present), "missing": len(missing),
        "ratio": ratio, "threshold": threshold, "warn": warn,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions",
                    default=str(Path(__file__).resolve().parents[1]
                                / "raw_data" / "positions_v2.jsonl"))
    ap.add_argument("--n-positions",            type=int,   default=1000)
    ap.add_argument("--questions-per-position", type=int,   default=3)
    ap.add_argument("--seed",                   type=int,   default=42)
    ap.add_argument("--base-ratio",             type=float, default=3.0,
                    help="Base max/min ratio for 13 classes; scales as sqrt(C/13)")
    ap.add_argument("--head",                   type=int,   default=5,
                    help="Top/bottom N tokens to print per QT")
    ap.add_argument("--question-types", nargs="*", default=None)
    ap.add_argument("--pov", choices=["abs", "rel", "both"], default="abs")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero on any WARN (default: report only)")
    args = ap.parse_args()

    if not os.path.exists(args.positions):
        sys.exit(f"positions file not found: {args.positions}")

    qts  = args.question_types or list(QUESTION_REGISTRY)
    povs = {"abs": [False], "rel": [True], "both": [False, True]}[args.pov]

    results: list[dict] = []
    for qt in qts:
        for pov in povs:
            counts, n_recs, n_unexp = measure(
                qt, pov, args.positions,
                args.n_positions, args.questions_per_position, args.seed,
            )
            expected = _expected_answer_classes(qt, pov)
            results.append(_print_distribution(
                qt, pov, counts, expected, n_recs, n_unexp,
                args.base_ratio, args.head,
            ))

    # ----- summary -----
    print("\n" + "=" * 78)
    print(f"  {'question_type':<30} {'enc':<4} {'present':<10} {'ratio':<8} "
          f"{'thr':<6} status")
    print("  " + "-" * 74)
    n_warn = 0
    for r in results:
        status = "WARN" if r["warn"] else "OK"
        if r["warn"]:
            n_warn += 1
        print(f"  {r['qt']:<30} {r['enc']:<4} "
              f"{r['present']:>3}/{r['n_classes']:<3}     "
              f"{r['ratio']:>6.2f}   {r['threshold']:>5.2f}   {status}")
    print("=" * 78)
    print(f"{n_warn}/{len(results)} (qt, enc) flagged as WARN")
    if args.strict and n_warn:
        sys.exit(1)


if __name__ == "__main__":
    main()
