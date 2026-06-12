"""Task: piece_on_square — name the piece (or EMPTY) on a queried square.

Self-contained: entity enumeration + frequency-aware weighter + build_qa all
inline. Per plans/merge_data_pipelines.md, no shared TaskSpec/weighter — a
bug in one task's weighter is fixed in that file alone.
"""
import random
from collections import defaultdict

from utils.board_representation import BoardRepr
from utils.utils import EMPTY_TOKEN

NAME = "piece_on_square"

SQ_QUESTIONS_TOK = [
    "What piece is on {sq_tok}?",
    "In this position, what piece occupies {sq_tok}?",
    "{sq_tok} contains what piece?",
]

SQ_OCCUPIED_ANSWERS_TOK = [
    "There is {piece_tok} on {sq_tok}.",
    "{sq_tok} contains {piece_tok}.",
    "In this position, {piece_tok} is on {sq_tok}.",
]

SQ_EMPTY_ANSWERS_TOK = [
    "{sq_tok} is empty.",
    "There is no piece on {sq_tok}.",
]


def sample_one(board: BoardRepr, frequency: dict, rng: random.Random) -> dict:
    # Entities: all 64 board squares. Two independent shaping factors,
    # multiplied to form the per-entity weight:
    #   (1) piece-class balance — mult correction + piece_tok freq, so each
    #       of the 13 answer classes (12 pieces + EMPTY) sees uniform rate.
    #   (2) queried-square balance — sq_tok freq, so each of the 64 squares
    #       is queried at uniform rate.
    entities   = list(range(64))
    piece_toks = [board.piece_at(sq) for sq in entities]
    sq_toks    = [board.sq_tok(sq)   for sq in entities]

    mult: dict[str, int] = defaultdict(int)
    for p in piece_toks:
        mult[p] += 1
    weights = [
        1.0 / (
            (frequency.get(p, 0) + 1) * mult[p]
            * (frequency.get(s, 0) + 1)
        )
        for p, s in zip(piece_toks, sq_toks)
    ]
    sq = rng.choices(entities, weights=weights, k=1)[0]

    piece_tok = board.piece_at(sq)
    sq_tok    = board.sq_tok(sq)
    if piece_tok == EMPTY_TOKEN:
        fmt = {"sq_tok": sq_tok}
        q = rng.choice(SQ_QUESTIONS_TOK).format(**fmt)
        a = rng.choice(SQ_EMPTY_ANSWERS_TOK).format(**fmt)
    else:
        fmt = {"sq_tok": sq_tok, "piece_tok": piece_tok}
        q = rng.choice(SQ_QUESTIONS_TOK).format(**fmt)
        a = rng.choice(SQ_OCCUPIED_ANSWERS_TOK).format(**fmt)

    parse_tag    = f"{sq_tok}{piece_tok}"
    answer_class = [sq_tok, piece_tok]
    return {
        "question":      q,
        "answer":        f"{a}\n\n{parse_tag}",
        "question_type": NAME,
        "answer_class":  answer_class,
    }
