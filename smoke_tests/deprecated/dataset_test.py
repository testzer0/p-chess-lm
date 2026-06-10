"""Comprehensive format & consistency smoke test for v2.1 and v3 SFT datasets.

Verifies every example in the small generated datasets against the spec in
plans/chess_plan.md:

  v2.1: pov=False, new_tok_in_query=True
        - SQUARE_TOKENS (<SQUARE_A1>...<SQUARE_H8>) and PIECE_TOKENS
        - tokens appear in BOTH question and answer prose
        - question_type ∈ {static_square, static_piece}
        - answer_class consistent with FEN (board-absolute)

  v3:   pov=True,  new_tok_in_query=True
        - POV_SQUARE_TOKENS (<SQUARE_1>...<SQUARE_64>) and PIECE_TOKENS
        - tokens appear in BOTH question and answer prose
        - question_type ∈ {static_square_pov, static_piece_pov}
        - answer_class consistent with FEN under POV mapping
          (white-to-move: pov_idx i == board sq i;
           black-to-move: pov_idx i == board sq i^56)

Usage
-----
source /scratch/gpfs/DANQIC/jeff/chesslm/.venv/bin/activate
python -m chesslm.smoke_tests.dataset_test \
    --v21-root chesslm/datasets/v2.1 \
    --v3-root  chesslm/datasets/v3   \
    --decoder-path /scratch/gpfs/DANQIC/jeff/models/smollm-3b-instruct

This will run a FULL scan of every example in {train, eval} for each dataset.
Pass --max-samples N to cap per-split sampling.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import chess
from datasets import load_from_disk
from transformers import AutoTokenizer

from chesslm.utils.utils import (
    ANSWER_SPECIAL_TOKENS,
    POV_ANSWER_SPECIAL_TOKENS,
    PIECE_TOKENS,
    POV_SQUARE_TOKENS,
    SQUARE_TOKENS,
    EMPTY_TOKEN,
)
from chesslm.utils.eval_utils import _is_valid_parse_tag, _is_consistent

EXPECTED_COLUMNS = {"question", "answer", "question_type", "answer_class",
                    "fen", "start_fen", "moves"}

# Per-variant configuration (what each dataset MUST look like).
VARIANTS = {
    "v2.1": {
        "pov": False,
        "new_tok_in_query": True,
        "expected_qts": {"static_square", "static_piece"},
        "sq_tokens": set(SQUARE_TOKENS),
        "wrong_sq_tokens": set(POV_SQUARE_TOKENS),
        "special_tokens": ANSWER_SPECIAL_TOKENS,
    },
    "v3": {
        "pov": True,
        "new_tok_in_query": True,
        "expected_qts": {"static_square_pov", "static_piece_pov"},
        "sq_tokens": set(POV_SQUARE_TOKENS),
        "wrong_sq_tokens": set(SQUARE_TOKENS),
        "special_tokens": POV_ANSWER_SPECIAL_TOKENS,
    },
}

# Regex matching ANY square-token shape (board-absolute OR POV) so we can
# detect leaks of the wrong variant cleanly even when both variants share the
# `<SQUARE_` prefix.
_SQUARE_TOK_RE = re.compile(r"<SQUARE_(?:[A-H][1-8]|\d{1,2})>")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(dataset_root: Path) -> dict:
    cfg_path = dataset_root / "dataset_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing {cfg_path}")
    with open(cfg_path) as f:
        return json.load(f)


def _classify_sq_token(tok: str) -> str:
    """'abs' if <SQUARE_E4>-style, 'pov' if <SQUARE_7>-style, else 'unknown'."""
    if not _SQUARE_TOK_RE.fullmatch(tok):
        return "unknown"
    inner = tok[len("<SQUARE_"):-1]
    return "abs" if inner[0].isalpha() else "pov"


def _extract_parse_tag_tokens(answer: str, special_token_set: set[str]) -> list[str]:
    parts = answer.split("\n\n")
    tag_text = parts[-1].strip() if len(parts) > 1 else ""
    pattern = "|".join(re.escape(t) for t in sorted(special_token_set, key=len, reverse=True))
    return re.findall(pattern, tag_text)


def _all_square_tokens_in(text: str) -> list[str]:
    return _SQUARE_TOK_RE.findall(text)


def _piece_tokens_in(text: str) -> list[str]:
    return [t for t in PIECE_TOKENS if t in text]


from chesslm.utils.utils import _PIECE_TO_TOKEN


def _pov_idx_from_sq_tok(sq_tok: str) -> int | None:
    if not _SQUARE_TOK_RE.fullmatch(sq_tok):
        return None
    inner = sq_tok[len("<SQUARE_"):-1]
    if inner[0].isalpha():
        return None
    return int(inner) - 1


def _piece_tok_at_pov(board: chess.Board, pov_idx: int) -> str:
    board_sq = pov_idx ^ 56 if board.turn == chess.BLACK else pov_idx
    piece = board.piece_at(board_sq)
    if piece is None:
        return EMPTY_TOKEN
    return _PIECE_TO_TOKEN[(piece.color, piece.piece_type)]


def _check_pov_idx_matches_board(fen: str, sq_tok: str, piece_tok: str) -> bool:
    """static_square_pov: <SQUARE_N> → piece at pov_idx N-1 (with sq^56 for black)."""
    pov_idx = _pov_idx_from_sq_tok(sq_tok)
    if pov_idx is None:
        return False
    return _piece_tok_at_pov(chess.Board(fen), pov_idx) == piece_tok


def _check_pov_piece_matches_board(fen: str, piece_tok: str, sq_toks: list[str]) -> bool:
    """static_piece_pov: piece_tok → exactly the listed pov idxs (or [EMPTY] if absent).

    Independent re-derivation: enumerate all 64 board squares, take pieces of
    matching (color, type), convert their board sq → pov_idx via XOR-56 when
    black to move, and compare to the answer_class set.
    """
    board = chess.Board(fen)
    expected_idxs = set()
    for board_sq in chess.SQUARES:
        piece = board.piece_at(board_sq)
        if piece is None:
            continue
        if _PIECE_TO_TOKEN[(piece.color, piece.piece_type)] != piece_tok:
            continue
        pov_idx = board_sq ^ 56 if board.turn == chess.BLACK else board_sq
        expected_idxs.add(pov_idx)

    if not expected_idxs:
        return sq_toks == [EMPTY_TOKEN]

    got_idxs = set()
    for t in sq_toks:
        idx = _pov_idx_from_sq_tok(t)
        if idx is None:
            return False
        got_idxs.add(idx)
    return got_idxs == expected_idxs


# ---------------------------------------------------------------------------
# Per-split checks
# ---------------------------------------------------------------------------

def check_split(name: str, ds, variant_cfg: dict, max_samples: int | None,
                tokenizer, fail_examples: list, indent: str = "  ") -> dict:
    """Returns a dict of pass/fail counters."""
    pov               = variant_cfg["pov"]
    expected_qts      = variant_cfg["expected_qts"]
    sq_tokens         = variant_cfg["sq_tokens"]
    wrong_sq_tokens   = variant_cfg["wrong_sq_tokens"]
    special_tokens    = variant_cfg["special_tokens"]
    special_set       = set(special_tokens)

    n_total = len(ds)
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    counters = Counter()
    qt_counts = Counter()
    per_qt_first_fail = {}

    def record_fail(reason: str, ex: dict, idx: int):
        if len(fail_examples) < 30:
            fail_examples.append({"split": name, "idx": idx, "reason": reason, "ex": ex})

    # Tokenizer check on a small subsample
    tok_split_bad = set()
    tok_check_indices = set(range(min(40, n_total)))

    for i in range(n_total):
        ex          = ds[i]
        qt          = ex["question_type"]
        fen         = ex["fen"]
        ac          = ex["answer_class"]
        question    = ex["question"]
        answer      = ex["answer"]
        start_fen   = ex["start_fen"]
        moves       = ex["moves"]

        qt_counts[qt] += 1
        counters["total"] += 1

        # 1. Columns / schema
        missing_cols = EXPECTED_COLUMNS - set(ex.keys())
        if missing_cols:
            record_fail(f"missing columns {missing_cols}", ex, i)

        # 2. FEN parseable
        try:
            board = chess.Board(fen)
            counters["fen_valid"] += 1
        except Exception:
            counters["fen_invalid"] += 1
            record_fail("invalid FEN", ex, i)
            continue

        # 3. start_fen parseable + moves replay to fen
        try:
            b2 = chess.Board(start_fen)
            for mv in moves:
                b2.push_uci(mv)
            if b2.fen() == fen:
                counters["start_fen_replay_ok"] += 1
            else:
                counters["start_fen_replay_mismatch"] += 1
                record_fail(f"start_fen+moves != fen ({b2.fen()})", ex, i)
        except Exception as e:
            counters["start_fen_replay_err"] += 1
            record_fail(f"start_fen replay error: {e}", ex, i)

        # 4. question_type matches variant
        if qt in expected_qts:
            counters["qt_ok"] += 1
        else:
            counters["qt_wrong"] += 1
            record_fail(f"unexpected qt {qt!r} for variant", ex, i)

        # 5. answer_class is a list of valid tokens (well-formed parse tag)
        if _is_valid_parse_tag(ac, qt):
            counters["parse_tag_valid"] += 1
        else:
            counters["parse_tag_invalid"] += 1
            if qt not in per_qt_first_fail:
                per_qt_first_fail[qt] = ("parse_tag_invalid", i)
            record_fail("invalid parse tag", ex, i)

        # 6. answer_class consistent with FEN
        if _is_consistent(qt, ac, fen):
            counters["fen_consistent"] += 1
        else:
            counters["fen_inconsistent"] += 1
            record_fail("answer_class not FEN-consistent", ex, i)

        # 7. answer parse tag (text-extracted) matches answer_class
        extracted = _extract_parse_tag_tokens(answer, special_set)
        if extracted == ac:
            counters["answer_text_matches"] += 1
        else:
            counters["answer_text_mismatch"] += 1
            record_fail(f"answer text tag {extracted} != answer_class {ac}", ex, i)

        # 8. Tokens-in-query check (v2.1/v3 both require tok_in_query=True)
        sq_in_q  = [t for t in _all_square_tokens_in(question) if t in sq_tokens]
        sq_in_a  = [t for t in _all_square_tokens_in(answer)   if t in sq_tokens]
        pc_in_q  = _piece_tokens_in(question)
        pc_in_a  = _piece_tokens_in(answer)

        # static_square_* questions reference a single square in the question.
        if qt in ("static_square", "static_square_pov"):
            if sq_in_q:
                counters["sq_tok_in_question"] += 1
            else:
                counters["sq_tok_in_question_missing"] += 1
                record_fail("static_square question lacks SQ token", ex, i)
            # Answer prose for empty/occupied should also contain the SQ token.
            if sq_in_a:
                counters["sq_tok_in_answer"] += 1
            else:
                counters["sq_tok_in_answer_missing"] += 1
                record_fail("static_square answer lacks SQ token in prose", ex, i)

        # static_piece_* questions reference a piece token in the question.
        if qt in ("static_piece", "static_piece_pov"):
            if pc_in_q:
                counters["pc_tok_in_question"] += 1
            else:
                counters["pc_tok_in_question_missing"] += 1
                record_fail("static_piece question lacks PIECE token", ex, i)
            # Answer either contains a PIECE_TOKEN (for present) or EMPTY (absent).
            # Parse-tag already validated; here we check prose has the squares
            # listed by token when present.
            if ac[1:] == [EMPTY_TOKEN]:
                # absent - prose doesn't list squares; OK to skip
                counters["pc_absent_ok"] += 1
            else:
                # present - prose should list every square token from the answer_class
                missing = [t for t in ac[1:] if t not in answer]
                if missing:
                    counters["pc_prose_missing_sq"] += 1
                    record_fail(f"piece-present prose missing {missing}", ex, i)
                else:
                    counters["pc_prose_ok"] += 1

        # 9. Wrong-variant token leakage
        wrong_q = [t for t in _all_square_tokens_in(question) if t in wrong_sq_tokens]
        wrong_a = [t for t in _all_square_tokens_in(answer)   if t in wrong_sq_tokens]
        if wrong_q or wrong_a:
            counters["wrong_variant_leak"] += 1
            record_fail(f"wrong-variant SQ tokens leaked q={wrong_q} a={wrong_a}", ex, i)
        else:
            counters["wrong_variant_clean"] += 1

        # 10. Independent POV semantics check (v3 only) — covers BOTH qt's
        if pov and qt == "static_square_pov":
            sq_tok, piece_tok = ac
            if _check_pov_idx_matches_board(fen, sq_tok, piece_tok):
                counters["pov_sq_semantics_ok"] += 1
            else:
                counters["pov_sq_semantics_bad"] += 1
                record_fail(f"POV sq idx mismatch: {sq_tok}->{piece_tok} on {fen}", ex, i)
        elif pov and qt == "static_piece_pov":
            piece_tok, sq_toks = ac[0], ac[1:]
            if _check_pov_piece_matches_board(fen, piece_tok, sq_toks):
                counters["pov_pc_semantics_ok"] += 1
            else:
                counters["pov_pc_semantics_bad"] += 1
                record_fail(f"POV piece mismatch: {piece_tok}->{sq_toks} on {fen}", ex, i)

        # 11. Tokenizer single-ID check
        if tokenizer is not None and i in tok_check_indices:
            for tok in special_set:
                if tok in question or tok in answer:
                    ids = tokenizer.encode(tok, add_special_tokens=False)
                    if len(ids) != 1:
                        tok_split_bad.add(tok)

    # Per-question-type counts
    counters_qt = {f"qt:{k}": v for k, v in qt_counts.items()}
    return {**counters, **counters_qt, "_tok_split_bad": tok_split_bad}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_variant(variant_name: str, dataset_root: Path, decoder_path: str | None,
                max_samples: int | None) -> bool:
    print(f"\n{'='*72}\n  Variant {variant_name}  root={dataset_root}\n{'='*72}")
    cfg = _load_config(dataset_root)
    variant_cfg = VARIANTS[variant_name]

    cfg_ok = (cfg.get("pov") == variant_cfg["pov"]
              and cfg.get("new_tok_in_query") == variant_cfg["new_tok_in_query"])
    print(f"  dataset_config.json: {cfg}  "
          f"({'OK' if cfg_ok else 'FAIL — does not match spec'})")
    if not cfg_ok:
        return False

    tokenizer = None
    if decoder_path:
        print(f"  loading tokenizer {decoder_path}")
        tokenizer = AutoTokenizer.from_pretrained(decoder_path, local_files_only=True)
        n_orig = len(tokenizer)
        tokenizer.add_tokens(variant_cfg["special_tokens"], special_tokens=True)
        added = len(tokenizer) - n_orig
        print(f"  tokenizer: added {added}/{len(variant_cfg['special_tokens'])} special tokens")
        if added != len(variant_cfg["special_tokens"]):
            print(f"  WARN: {len(variant_cfg['special_tokens']) - added} tokens already existed")

    fail_examples = []
    all_ok = True

    for split in ("train", "eval"):
        split_path = dataset_root / split
        if not split_path.exists():
            print(f"\n  [{split}] missing — skipping")
            continue
        print(f"\n  [{split}] loading {split_path}")
        ds = load_from_disk(str(split_path))
        print(f"  [{split}] {len(ds):,} examples  columns: {ds.column_names}")

        missing = EXPECTED_COLUMNS - set(ds.column_names)
        if missing:
            print(f"  [{split}] FAIL missing columns: {missing}")
            all_ok = False
            continue

        c = check_split(split, ds, variant_cfg, max_samples, tokenizer,
                        fail_examples=fail_examples)

        n = c["total"]
        def line(label, key, fail_key=None):
            ok_n = c.get(key, 0)
            fail_n = c.get(fail_key, 0) if fail_key else (n - ok_n)
            status = "OK" if fail_n == 0 else f"FAIL ({fail_n})"
            print(f"    {label:<34} {ok_n}/{n}   {status}")
            return fail_n == 0

        split_ok = True
        split_ok &= line("FEN valid",                "fen_valid",            "fen_invalid")
        split_ok &= line("start_fen+moves replay",   "start_fen_replay_ok",  "start_fen_replay_mismatch")
        split_ok &= line("question_type matches",    "qt_ok",                "qt_wrong")
        split_ok &= line("parse_tag valid",          "parse_tag_valid",      "parse_tag_invalid")
        split_ok &= line("FEN consistent",           "fen_consistent",       "fen_inconsistent")
        split_ok &= line("answer text == class",     "answer_text_matches",  "answer_text_mismatch")
        split_ok &= line("no wrong-variant SQ tok",  "wrong_variant_clean",  "wrong_variant_leak")

        if variant_cfg["pov"]:
            line("POV sq->piece semantics",     "pov_sq_semantics_ok",  "pov_sq_semantics_bad")
            line("POV piece->squares semantics","pov_pc_semantics_ok",  "pov_pc_semantics_bad")

        # Question type breakdown
        print(f"    question_type distribution:")
        for k, v in sorted(c.items()):
            if isinstance(k, str) and k.startswith("qt:"):
                print(f"      {k[3:]:<28} {v}")

        if c.get("_tok_split_bad"):
            print(f"    [WARN] tokenizer splits these tokens into >1 ID: {sorted(c['_tok_split_bad'])}")
            split_ok = False
        elif tokenizer is not None:
            print(f"    tokenizer: all special tokens encode as single IDs (subsample)")

        all_ok &= split_ok

    if fail_examples:
        print(f"\n  ---- First failing examples ({len(fail_examples)}) ----")
        for f in fail_examples[:8]:
            ex = f["ex"]
            print(f"\n  [{f['split']} idx={f['idx']}] {f['reason']}")
            print(f"    qt={ex['question_type']}  fen={ex['fen']}")
            print(f"    Q={ex['question']}")
            print(f"    A={ex['answer']}")
            print(f"    class={ex['answer_class']}")

    return all_ok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--v21-root", default="chesslm/datasets/v2.1")
    p.add_argument("--v3-root",  default="chesslm/datasets/v3")
    p.add_argument("--decoder-path", default=None,
                   help="Optional path to SmolLM3 tokenizer for token-ID checks")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap samples per split (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    ok21 = run_variant("v2.1", Path(args.v21_root), args.decoder_path, args.max_samples)
    ok3  = run_variant("v3",   Path(args.v3_root),  args.decoder_path, args.max_samples)

    print("\n" + "=" * 72)
    print(f"  v2.1: {'PASS' if ok21 else 'FAIL'}")
    print(f"  v3:   {'PASS' if ok3  else 'FAIL'}")
    print("=" * 72)
    sys.exit(0 if (ok21 and ok3) else 1)


if __name__ == "__main__":
    main()
