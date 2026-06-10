"""Generate SFT QA pairs from chess positions.

question_type variants
----------------------
  static_square     — board-absolute, plain or tok-in-query
  static_piece      — board-absolute, plain or tok-in-query
  static_square_pov — POV-relative, always tok-in-query
  static_piece_pov  — POV-relative, always tok-in-query

new_tok_in_query
----------------
  False (v2):  plain-text square/piece names in question AND answer prose
  True (v2.1): special tokens in question AND answer prose
  (always True for _pov types)
"""
import argparse
import chess
import json
import random
from collections import defaultdict
from pathlib import Path

from chesslm.utils.prompt_utils import (
    SQ_QUESTIONS,     SQ_OCCUPIED_ANSWERS,     SQ_EMPTY_ANSWERS,
    SQ_QUESTIONS_TOK, SQ_OCCUPIED_ANSWERS_TOK, SQ_EMPTY_ANSWERS_TOK,
    PC_QUESTIONS,     PC_PRESENT_ANSWERS,       PC_ABSENT_ANSWERS,
    PC_QUESTIONS_TOK, PC_PRESENT_ANSWERS_TOK,   PC_ABSENT_ANSWERS_TOK,
)
from chesslm.utils.utils import (
    ANSWER_SPECIAL_TOKENS,
    EMPTY_TOKEN,
    PIECE_TOKENS,
    POV_SQUARE_TOKENS,
    SQUARE_TOKENS,
    _PIECE_TO_TOKEN,
    _generate_position_dict,
    _generate_pov_position_dict,
)

_PIECE_COLOR_NAME = {chess.WHITE: "white",  chess.BLACK: "black"}
_PIECE_TYPE_NAME  = {
    chess.PAWN:   "pawn",   chess.KNIGHT: "knight",
    chess.BISHOP: "bishop", chess.ROOK:   "rook",
    chess.QUEEN:  "queen",  chess.KING:   "king",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weighted_sample(candidates: list[str], frequency: dict, rng: random.Random) -> str:
    weights = [1.0 / (frequency.get(c, 0) + 1) for c in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


def _prose_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _fill_qa(q_tmpls, a_tmpls, fmt_vars, parse_tag, answer_class, rng):
    q = rng.choice(q_tmpls).format(**fmt_vars)
    a = rng.choice(a_tmpls).format(**fmt_vars)
    return q, f"{a}\n\n{parse_tag}", answer_class


def _sq_tmpls(tok: bool):
    """Return (q_tmpls, occupied_tmpls, empty_tmpls) for the requested variant."""
    if tok:
        return SQ_QUESTIONS_TOK, SQ_OCCUPIED_ANSWERS_TOK, SQ_EMPTY_ANSWERS_TOK
    return SQ_QUESTIONS, SQ_OCCUPIED_ANSWERS, SQ_EMPTY_ANSWERS


def _pc_tmpls(tok: bool):
    """Return (q_tmpls, present_tmpls, absent_tmpls) for the requested variant."""
    if tok:
        return PC_QUESTIONS_TOK, PC_PRESENT_ANSWERS_TOK, PC_ABSENT_ANSWERS_TOK
    return PC_QUESTIONS, PC_PRESENT_ANSWERS, PC_ABSENT_ANSWERS


# ---------------------------------------------------------------------------
# Board-absolute sampling helpers (v2 / v2.1)
# ---------------------------------------------------------------------------

def _static_square(square_to_piece, frequency, rng, tok: bool = False):
    class_counts = defaultdict(int)
    for t in square_to_piece.values():
        class_counts[t] += 1

    squares = list(square_to_piece)
    weights = [
        (1.0 / (frequency.get(square_to_piece[sq], 0) + 1)) / class_counts[square_to_piece[sq]]
        for sq in squares
    ]
    chosen_sq = rng.choices(squares, weights=weights, k=1)[0]
    piece_tok = square_to_piece[chosen_sq]
    sq_tok    = SQUARE_TOKENS[chess.parse_square(chosen_sq)]
    q_t, occ_t, emp_t = _sq_tmpls(tok)

    if piece_tok == EMPTY_TOKEN:
        fmt = {"square": chosen_sq, "sq_tok": sq_tok}
        return q_t, emp_t, fmt, f"{sq_tok}{piece_tok}", [sq_tok, piece_tok]

    color, ptype = next(k for k, v in _PIECE_TO_TOKEN.items() if v == piece_tok)
    fmt = {"color": _PIECE_COLOR_NAME[color], "piece": _PIECE_TYPE_NAME[ptype],
           "square": chosen_sq, "sq_tok": sq_tok, "piece_tok": piece_tok}
    return q_t, occ_t, fmt, f"{sq_tok}{piece_tok}", [sq_tok, piece_tok]


def _static_piece(piece_to_squares, frequency, rng, tok: bool = False):
    n_absent   = sum(1 for t in PIECE_TOKENS if not piece_to_squares[t])
    empty_freq = frequency.get(EMPTY_TOKEN, 0) * max(1, n_absent)

    piece_freq = {
        t: (sum(frequency.get(SQUARE_TOKENS[chess.parse_square(s)], 0) for s in piece_to_squares[t])
            if piece_to_squares[t] else empty_freq)
        for t in PIECE_TOKENS
    }
    piece_tok  = _weighted_sample(PIECE_TOKENS, piece_freq, rng)
    color, ptype = next(k for k, v in _PIECE_TO_TOKEN.items() if v == piece_tok)
    fmt_base   = {"color": _PIECE_COLOR_NAME[color], "piece": _PIECE_TYPE_NAME[ptype],
                  "piece_tok": piece_tok}
    squares    = piece_to_squares[piece_tok]
    q_t, pre_t, abs_t = _pc_tmpls(tok)

    if not squares:
        return q_t, abs_t, fmt_base, f"{piece_tok}{EMPTY_TOKEN}", [piece_tok, EMPTY_TOKEN]

    sq_toks  = [SQUARE_TOKENS[chess.parse_square(s)] for s in squares]
    prose    = _prose_list(sq_toks if tok else squares)
    return q_t, pre_t, {**fmt_base, "squares": prose}, f"{piece_tok}{''.join(sq_toks)}", [piece_tok] + sq_toks


# ---------------------------------------------------------------------------
# POV-relative sampling helpers (v3) — always tok
# ---------------------------------------------------------------------------

def _static_square_pov(pov_idx_to_piece, frequency, rng):
    class_counts = defaultdict(int)
    for t in pov_idx_to_piece.values():
        class_counts[t] += 1

    indices = list(pov_idx_to_piece)
    weights = [
        (1.0 / (frequency.get(pov_idx_to_piece[i], 0) + 1)) / class_counts[pov_idx_to_piece[i]]
        for i in indices
    ]
    chosen_idx = rng.choices(indices, weights=weights, k=1)[0]
    piece_tok  = pov_idx_to_piece[chosen_idx]
    sq_tok     = POV_SQUARE_TOKENS[chosen_idx]

    if piece_tok == EMPTY_TOKEN:
        fmt = {"sq_tok": sq_tok}
        return SQ_QUESTIONS_TOK, SQ_EMPTY_ANSWERS_TOK, fmt, f"{sq_tok}{piece_tok}", [sq_tok, piece_tok]

    color, ptype = next(k for k, v in _PIECE_TO_TOKEN.items() if v == piece_tok)
    fmt = {"color": _PIECE_COLOR_NAME[color], "piece": _PIECE_TYPE_NAME[ptype],
           "sq_tok": sq_tok, "piece_tok": piece_tok}
    return SQ_QUESTIONS_TOK, SQ_OCCUPIED_ANSWERS_TOK, fmt, f"{sq_tok}{piece_tok}", [sq_tok, piece_tok]


def _static_piece_pov(piece_to_pov_idxs, frequency, rng):
    n_absent   = sum(1 for t in PIECE_TOKENS if not piece_to_pov_idxs[t])
    empty_freq = frequency.get(EMPTY_TOKEN, 0) * max(1, n_absent)

    piece_freq = {
        t: (sum(frequency.get(POV_SQUARE_TOKENS[i], 0) for i in piece_to_pov_idxs[t])
            if piece_to_pov_idxs[t] else empty_freq)
        for t in PIECE_TOKENS
    }
    piece_tok  = _weighted_sample(PIECE_TOKENS, piece_freq, rng)
    color, ptype = next(k for k, v in _PIECE_TO_TOKEN.items() if v == piece_tok)
    fmt_base   = {"color": _PIECE_COLOR_NAME[color], "piece": _PIECE_TYPE_NAME[ptype],
                  "piece_tok": piece_tok}
    idxs = piece_to_pov_idxs[piece_tok]

    if not idxs:
        return PC_QUESTIONS_TOK, PC_ABSENT_ANSWERS_TOK, fmt_base, f"{piece_tok}{EMPTY_TOKEN}", [piece_tok, EMPTY_TOKEN]

    sq_toks = [POV_SQUARE_TOKENS[i] for i in idxs]
    fmt = {**fmt_base, "squares": _prose_list(sq_toks)}
    return PC_QUESTIONS_TOK, PC_PRESENT_ANSWERS_TOK, fmt, f"{piece_tok}{''.join(sq_toks)}", [piece_tok] + sq_toks


# ---------------------------------------------------------------------------
# Exhaustive eval generation
# ---------------------------------------------------------------------------

def generate_eval_set(
    fen: str,
    question_type: str,
    start_fen: str = "",
    moves: list = None,
    seed: int = 0,
    new_tok_in_query: bool = False,
) -> list[dict]:
    rng  = random.Random(seed)
    base = {"fen": fen, "start_fen": start_fen, "moves": moves or [], "question_type": question_type}
    tok  = new_tok_in_query
    results = []

    match question_type:

        case "static_square":
            square_to_piece, _ = _generate_position_dict(fen)
            q_t, occ_t, emp_t  = _sq_tmpls(tok)
            for sq in chess.SQUARES:
                sq_name   = chess.square_name(sq)
                piece_tok = square_to_piece[sq_name]
                sq_tok    = SQUARE_TOKENS[sq]
                if piece_tok == EMPTY_TOKEN:
                    fmt  = {"square": sq_name, "sq_tok": sq_tok}
                    args = (q_t, emp_t, fmt, f"{sq_tok}{piece_tok}", [sq_tok, piece_tok])
                else:
                    color, ptype = next(k for k, v in _PIECE_TO_TOKEN.items() if v == piece_tok)
                    fmt  = {"color": _PIECE_COLOR_NAME[color], "piece": _PIECE_TYPE_NAME[ptype],
                            "square": sq_name, "sq_tok": sq_tok, "piece_tok": piece_tok}
                    args = (q_t, occ_t, fmt, f"{sq_tok}{piece_tok}", [sq_tok, piece_tok])
                q, answer, answer_class = _fill_qa(*args, rng)
                results.append({**base, "question": q, "answer": answer, "answer_class": answer_class})

        case "static_piece":
            _, piece_to_squares = _generate_position_dict(fen)
            q_t, pre_t, abs_t  = _pc_tmpls(tok)
            for piece_tok in PIECE_TOKENS:
                color, ptype = next(k for k, v in _PIECE_TO_TOKEN.items() if v == piece_tok)
                fmt_base = {"color": _PIECE_COLOR_NAME[color], "piece": _PIECE_TYPE_NAME[ptype],
                            "piece_tok": piece_tok}
                squares  = piece_to_squares[piece_tok]
                if not squares:
                    args = (q_t, abs_t, fmt_base, f"{piece_tok}{EMPTY_TOKEN}", [piece_tok, EMPTY_TOKEN])
                else:
                    sq_toks = [SQUARE_TOKENS[chess.parse_square(s)] for s in squares]
                    prose   = _prose_list(sq_toks if tok else squares)
                    args = (q_t, pre_t, {**fmt_base, "squares": prose},
                            f"{piece_tok}{''.join(sq_toks)}", [piece_tok] + sq_toks)
                q, answer, answer_class = _fill_qa(*args, rng)
                results.append({**base, "question": q, "answer": answer, "answer_class": answer_class})

        case "static_square_pov":
            pov_idx_to_piece, _ = _generate_pov_position_dict(fen)
            for pov_idx in range(64):
                piece_tok = pov_idx_to_piece[pov_idx]
                sq_tok    = POV_SQUARE_TOKENS[pov_idx]
                if piece_tok == EMPTY_TOKEN:
                    fmt  = {"sq_tok": sq_tok}
                    args = (SQ_QUESTIONS_TOK, SQ_EMPTY_ANSWERS_TOK, fmt,
                            f"{sq_tok}{piece_tok}", [sq_tok, piece_tok])
                else:
                    color, ptype = next(k for k, v in _PIECE_TO_TOKEN.items() if v == piece_tok)
                    fmt  = {"color": _PIECE_COLOR_NAME[color], "piece": _PIECE_TYPE_NAME[ptype],
                            "sq_tok": sq_tok, "piece_tok": piece_tok}
                    args = (SQ_QUESTIONS_TOK, SQ_OCCUPIED_ANSWERS_TOK, fmt,
                            f"{sq_tok}{piece_tok}", [sq_tok, piece_tok])
                q, answer, answer_class = _fill_qa(*args, rng)
                results.append({**base, "question": q, "answer": answer, "answer_class": answer_class})

        case "static_piece_pov":
            _, piece_to_pov_idxs = _generate_pov_position_dict(fen)
            for piece_tok in PIECE_TOKENS:
                color, ptype = next(k for k, v in _PIECE_TO_TOKEN.items() if v == piece_tok)
                fmt_base = {"color": _PIECE_COLOR_NAME[color], "piece": _PIECE_TYPE_NAME[ptype],
                            "piece_tok": piece_tok}
                idxs = piece_to_pov_idxs[piece_tok]
                if not idxs:
                    args = (PC_QUESTIONS_TOK, PC_ABSENT_ANSWERS_TOK, fmt_base,
                            f"{piece_tok}{EMPTY_TOKEN}", [piece_tok, EMPTY_TOKEN])
                else:
                    sq_toks = [POV_SQUARE_TOKENS[i] for i in idxs]
                    args = (PC_QUESTIONS_TOK, PC_PRESENT_ANSWERS_TOK,
                            {**fmt_base, "squares": _prose_list(sq_toks)},
                            f"{piece_tok}{''.join(sq_toks)}", [piece_tok] + sq_toks)
                q, answer, answer_class = _fill_qa(*args, rng)
                results.append({**base, "question": q, "answer": answer, "answer_class": answer_class})

        case _:
            raise ValueError(f"Unknown question_type: {question_type!r}")

    return results


# ---------------------------------------------------------------------------
# Core sampling function
# ---------------------------------------------------------------------------

def sample_question_from_position(
    fen: str,
    question_type: str,
    frequency: dict | None = None,
    rng: random.Random | None = None,
    new_tok_in_query: bool = False,
) -> dict:
    if rng is None:
        rng = random
    if frequency is None:
        frequency = {}

    tok = new_tok_in_query

    match question_type:
        case "static_square":
            square_to_piece, _ = _generate_position_dict(fen)
            args = _static_square(square_to_piece, frequency, rng, tok)
        case "static_piece":
            _, piece_to_squares = _generate_position_dict(fen)
            args = _static_piece(piece_to_squares, frequency, rng, tok)
        case "static_square_pov":
            pov_idx_to_piece, _ = _generate_pov_position_dict(fen)
            args = _static_square_pov(pov_idx_to_piece, frequency, rng)
        case "static_piece_pov":
            _, piece_to_pov_idxs = _generate_pov_position_dict(fen)
            args = _static_piece_pov(piece_to_pov_idxs, frequency, rng)
        case _:
            raise ValueError(f"Unknown question_type: {question_type!r}")

    q, answer, answer_class = _fill_qa(*args, rng)
    return {
        "question":      q,
        "answer":        answer,
        "question_type": question_type,
        "answer_class":  answer_class,
        "fen":           fen,
    }


# ---------------------------------------------------------------------------
# CLI (standalone generation to JSONL)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="chesslm/raw_data/positions.jsonl")
    parser.add_argument("--output", default="chesslm/data/sft_data.jsonl")
    parser.add_argument("--question-type", default="all",
                        choices=["static_square", "static_piece",
                                 "static_square_pov", "static_piece_pov", "all"])
    parser.add_argument("--n-positions", type=int, default=-1)
    parser.add_argument("--questions-per-position", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng  = random.Random(args.seed)
    question_types = (
        ["static_square", "static_piece"] if args.question_type == "all"
        else [args.question_type]
    )
    frequency: dict[str, dict[str, int]] = {qt: defaultdict(int) for qt in question_types}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for i, line in enumerate(fin):
            if args.n_positions >= 0 and i >= args.n_positions:
                break
            start_fen, moves, end_fen = json.loads(line)
            for _ in range(args.questions_per_position):
                qt     = rng.choice(question_types)
                sample = sample_question_from_position(
                    fen=end_fen, question_type=qt, frequency=frequency[qt], rng=rng,
                )
                sample["start_fen"] = start_fen
                sample["moves"]     = moves
                for cls in sample["answer_class"]:
                    frequency[qt][cls] += 1
                fout.write(json.dumps(sample) + "\n")
                n_written += 1

    print(f"Wrote {n_written} QA pairs to {args.output}")


if __name__ == "__main__":
    main()
