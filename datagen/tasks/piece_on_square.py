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
    "There is a {piece_tok} on {sq_tok}.",
    "{sq_tok} contains a {piece_tok}.",
    "In this position, a {piece_tok} is on {sq_tok}.",
]

SQ_EMPTY_ANSWERS_TOK = [
    "{sq_tok} is empty.",
    "There are no pieces on {sq_tok}.",
]


def sample_one(board: BoardRepr, frequency: dict, rng: random.Random) -> dict:
    # Entities: all 64 board squares. Answer-side weighting tokens: just the
    # piece on the square (sq_tok is 1-1 with entity choice so it doesn't help
    # balance). Multiplicity correction pulls back the ~32 empty squares so
    # the answer class sees uniform expected rate.
    entities = list(range(64))
    answer_tuples = [(board.piece_at(sq),) for sq in entities]

    mult: dict[tuple, int] = defaultdict(int)
    for a in answer_tuples:
        mult[a] += 1
    weights = [
        1.0 / ((sum(frequency.get(t, 0) for t in a) + 1) * mult[a])
        for a in answer_tuples
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
