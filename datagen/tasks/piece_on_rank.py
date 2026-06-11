"""Task: piece_on_rank — direct & CoT variants of "what pieces are on this rank?".

Copy-paste twin of piece_on_file.py with files() -> ranks(). Per
plans/merge_data_pipelines.md, the duplication is the cost of per-task
isolation — a bug here doesn't propagate to the other group tasks.
"""
import random
from collections import defaultdict

from utils.board_representation import BoardRepr
from datagen.prose import format_piece_counts, format_square_breakdown
from utils.utils import EMPTY_TOKEN

NAME_DIRECT = "piece_on_rank_direct"
NAME_COT    = "piece_on_rank_cot"

# TODO(nl-prose): DUMMY — 1 variant per list. Expand before production runs;
# render samples end-to-end and read them out loud to catch grammar oddities.
RANK_QUESTIONS_TOK = [
    "What pieces are on the rank from {start_tok} to {end_tok}?",
]

RANK_DIRECT_PRESENT_ANSWERS_TOK = [
    "There are {piece_counts} on the rank from {start_tok} to {end_tok}.",
]

RANK_COT_PRESENT_ANSWERS_TOK = [
    "Walking the rank from {start_tok} to {end_tok}: {square_breakdown}.",
]

RANK_EMPTY_ANSWERS_TOK = [
    "There are no pieces on the rank from {start_tok} to {end_tok}. It is an empty rank.",
]


def _answer_tuple(rank_sqs: tuple, board: BoardRepr) -> tuple:
    order = {tok: i for i, tok in enumerate(board.piece_tokens)}
    pieces = [board.piece_at(sq) for sq in rank_sqs
              if board.piece_at(sq) != EMPTY_TOKEN]
    if not pieces:
        return (EMPTY_TOKEN,)
    return tuple(sorted(pieces, key=order.__getitem__))


def _choose_rank(board: BoardRepr, frequency: dict, rng: random.Random) -> tuple:
    entities = board.ranks()
    answer_tuples = [_answer_tuple(e, board) for e in entities]

    mult: dict[tuple, int] = defaultdict(int)
    for a in answer_tuples:
        mult[a] += 1
    weights = [
        1.0 / ((sum(frequency.get(t, 0) for t in a) + 1) * mult[a])
        for a in answer_tuples
    ]
    return rng.choices(entities, weights=weights, k=1)[0]


def _piece_counts(rank_sqs: tuple, board: BoardRepr) -> list:
    counts: dict[str, int] = defaultdict(int)
    for sq in rank_sqs:
        p = board.piece_at(sq)
        if p != EMPTY_TOKEN:
            counts[p] += 1
    return [(p, counts[p]) for p in board.piece_tokens if counts[p] > 0]


def _common_facts(rank_sqs: tuple, board: BoardRepr) -> dict:
    start_tok = board.sq_tok(rank_sqs[0])
    end_tok   = board.sq_tok(rank_sqs[-1])
    ordered   = _piece_counts(rank_sqs, board)
    if not ordered:
        return {
            "start_tok":    start_tok,
            "end_tok":      end_tok,
            "ordered":      [],
            "parse_tag":    EMPTY_TOKEN,
            "answer_class": [start_tok, end_tok, EMPTY_TOKEN],
        }
    parse_tag    = "".join(p * c for p, c in ordered)
    pieces_flat  = [p for p, c in ordered for _ in range(c)]
    answer_class = [start_tok, end_tok] + pieces_flat
    return {
        "start_tok":    start_tok,
        "end_tok":      end_tok,
        "ordered":      ordered,
        "parse_tag":    parse_tag,
        "answer_class": answer_class,
    }


def sample_one_direct(board: BoardRepr, frequency: dict, rng: random.Random) -> dict:
    rank_sqs = _choose_rank(board, frequency, rng)
    f = _common_facts(rank_sqs, board)

    if not f["ordered"]:
        fmt = {"start_tok": f["start_tok"], "end_tok": f["end_tok"]}
        a_t = RANK_EMPTY_ANSWERS_TOK
    else:
        fmt = {
            "start_tok":    f["start_tok"],
            "end_tok":      f["end_tok"],
            "piece_counts": format_piece_counts(f["ordered"]),
        }
        a_t = RANK_DIRECT_PRESENT_ANSWERS_TOK
    q = rng.choice(RANK_QUESTIONS_TOK).format(**fmt)
    a = rng.choice(a_t).format(**fmt)
    return {
        "question":      q,
        "answer":        f"{a}\n\n{f['parse_tag']}",
        "question_type": NAME_DIRECT,
        "answer_class":  f["answer_class"],
    }


def sample_one_cot(board: BoardRepr, frequency: dict, rng: random.Random) -> dict:
    rank_sqs = _choose_rank(board, frequency, rng)
    f = _common_facts(rank_sqs, board)

    if not f["ordered"]:
        fmt = {"start_tok": f["start_tok"], "end_tok": f["end_tok"]}
        a_t = RANK_EMPTY_ANSWERS_TOK
    else:
        items = [
            (board.sq_tok(sq),
             None if board.piece_at(sq) == EMPTY_TOKEN else board.piece_at(sq))
            for sq in rank_sqs
        ]
        fmt = {
            "start_tok":        f["start_tok"],
            "end_tok":          f["end_tok"],
            "square_breakdown": format_square_breakdown(items),
        }
        a_t = RANK_COT_PRESENT_ANSWERS_TOK
    q = rng.choice(RANK_QUESTIONS_TOK).format(**fmt)
    a = rng.choice(a_t).format(**fmt)
    return {
        "question":      q,
        "answer":        f"{a}\n\n{f['parse_tag']}",
        "question_type": NAME_COT,
        "answer_class":  f["answer_class"],
    }
