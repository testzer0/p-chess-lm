"""Task: piece_on_file — direct & CoT variants of "what pieces are on this file?".

Two NAME constants and two sample_one_* functions. Direct and CoT share entity
enumeration + weighter (same answer space, same balancing target) and diverge
only in build_qa. Co-located so the diff is visible.

Per plans/merge_data_pipelines.md, no shared TaskSpec/weighter abstraction
across tasks. Group tasks (file / rank / diagonal) intentionally repeat the
same pattern in three files — same tradeoff.
"""
import random
from collections import defaultdict

from utils.board_representation import BoardRepr
from datagen.prose import format_piece_counts, format_square_breakdown
from utils.utils import EMPTY_TOKEN

NAME_DIRECT = "piece_on_file_direct"
NAME_COT    = "piece_on_file_cot"

# TODO(nl-prose): DUMMY — 1 variant per list. Expand before production runs;
# render samples end-to-end and read them out loud to catch grammar oddities.
FILE_QUESTIONS_TOK = [
    "What pieces are on the file from {start_tok} to {end_tok}?",
]

FILE_DIRECT_PRESENT_ANSWERS_TOK = [
    "There are {piece_counts} on the file from {start_tok} to {end_tok}.",
]

FILE_COT_PRESENT_ANSWERS_TOK = [
    "Walking the file from {start_tok} to {end_tok}: {square_breakdown}.",
]

FILE_EMPTY_ANSWERS_TOK = [
    "There are no pieces on the file from {start_tok} to {end_tok}. It is an open file.",
]


def _answer_tuple(file_sqs: tuple, board: BoardRepr) -> tuple:
    """Weighting tokens for one file: sorted multiset of piece tokens, or
    (EMPTY_TOKEN,) for an open file. Sort by canonical piece_tokens order so
    files holding the same multiset hash to the same tuple."""
    order = {tok: i for i, tok in enumerate(board.piece_tokens)}
    pieces = [board.piece_at(sq) for sq in file_sqs
              if board.piece_at(sq) != EMPTY_TOKEN]
    if not pieces:
        return (EMPTY_TOKEN,)
    return tuple(sorted(pieces, key=order.__getitem__))


def _choose_file(board: BoardRepr, frequency: dict, rng: random.Random) -> tuple:
    entities = board.files()
    answer_tuples = [_answer_tuple(e, board) for e in entities]

    mult: dict[tuple, int] = defaultdict(int)
    for a in answer_tuples:
        mult[a] += 1
    weights = [
        1.0 / ((sum(frequency.get(t, 0) for t in a) + 1) * mult[a])
        for a in answer_tuples
    ]
    return rng.choices(entities, weights=weights, k=1)[0]


def _piece_counts(file_sqs: tuple, board: BoardRepr) -> list:
    """[(piece_tok, n), ...] in canonical piece_tokens order, non-zero only."""
    counts: dict[str, int] = defaultdict(int)
    for sq in file_sqs:
        p = board.piece_at(sq)
        if p != EMPTY_TOKEN:
            counts[p] += 1
    return [(p, counts[p]) for p in board.piece_tokens if counts[p] > 0]


def _common_facts(file_sqs: tuple, board: BoardRepr) -> dict:
    start_tok = board.sq_tok(file_sqs[0])
    end_tok   = board.sq_tok(file_sqs[-1])
    ordered   = _piece_counts(file_sqs, board)
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
    file_sqs = _choose_file(board, frequency, rng)
    f = _common_facts(file_sqs, board)

    if not f["ordered"]:
        fmt = {"start_tok": f["start_tok"], "end_tok": f["end_tok"]}
        a_t = FILE_EMPTY_ANSWERS_TOK
    else:
        fmt = {
            "start_tok":    f["start_tok"],
            "end_tok":      f["end_tok"],
            "piece_counts": format_piece_counts(f["ordered"]),
        }
        a_t = FILE_DIRECT_PRESENT_ANSWERS_TOK
    q = rng.choice(FILE_QUESTIONS_TOK).format(**fmt)
    a = rng.choice(a_t).format(**fmt)
    return {
        "question":      q,
        "answer":        f"{a}\n\n{f['parse_tag']}",
        "question_type": NAME_DIRECT,
        "answer_class":  f["answer_class"],
    }


def sample_one_cot(board: BoardRepr, frequency: dict, rng: random.Random) -> dict:
    file_sqs = _choose_file(board, frequency, rng)
    f = _common_facts(file_sqs, board)

    if not f["ordered"]:
        fmt = {"start_tok": f["start_tok"], "end_tok": f["end_tok"]}
        a_t = FILE_EMPTY_ANSWERS_TOK
    else:
        items = [
            (board.sq_tok(sq),
             None if board.piece_at(sq) == EMPTY_TOKEN else board.piece_at(sq))
            for sq in file_sqs
        ]
        fmt = {
            "start_tok":        f["start_tok"],
            "end_tok":          f["end_tok"],
            "square_breakdown": format_square_breakdown(items),
        }
        a_t = FILE_COT_PRESENT_ANSWERS_TOK
    q = rng.choice(FILE_QUESTIONS_TOK).format(**fmt)
    a = rng.choice(a_t).format(**fmt)
    return {
        "question":      q,
        "answer":        f"{a}\n\n{f['parse_tag']}",
        "question_type": NAME_COT,
        "answer_class":  f["answer_class"],
    }
