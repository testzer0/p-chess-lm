"""Task: piece_on_diagonal — "what pieces are on this diagonal?"

Single task with two prose families (direct vs CoT) sampled per question.
Diagonal selection is uniform over the 26 diagonals (13 up-right + 13
up-left; both walk bottom-up). Compact piece-with-count encoding
`<PIECE>?N` in parse_tag + answer_class via
`datagen.prose.encode_piece_count`; reused across the line tasks and
piece_count. Grader: counted multiset (`utils.eval_utils._multiset_grade`).
"""
import random

from utils.board_representation import BoardRepr
from datagen.prose import (
    format_piece_counts,
    format_square_breakdown,
    line_facts,
)
from utils.utils import EMPTY_TOKEN

NAME = "piece_on_diagonal"

# Probability of routing a question through the CoT (per-square walk) template
# family vs the direct family. Hardcoded for now; lift to a CLI flag when
# build_qa_dataset.py gains task-level options.
COT_RATIO = 0.5

DIAGONAL_QUESTIONS_TOK = [
    "What piece(s) are on the diagonal from {start_tok} to {end_tok}?",
    "Identify the piece(s) that occupy the {start_tok}{end_tok} diagonal.",
    "In this position, the {start_tok}{end_tok} diagonal contains what pieces(s)?"
]

DIAGONAL_DIRECT_PRESENT_ANSWERS_TOK = [
    "There are {piece_counts} on the diagonal from {start_tok} to {end_tok}.",
    "The {start_tok}{end_tok} diagonal contains {piece_counts}.",
    "In this position, {piece_counts} occupy the {start_tok}{end_tok} diagonal."
]

DIAGONAL_COT_PRESENT_ANSWERS_TOK = [
    "Walking the diagonal from {start_tok} to {end_tok}: {square_breakdown}.",
    "Going up the {start_tok}{end_tok} diagonal: {square_breakdown}."
]

DIAGONAL_EMPTY_ANSWERS_TOK = [
    "There are no pieces on the diagonal from {start_tok} to {end_tok}. It is an open diagonal.",
    "The {start_tok}{end_tok} diagonal is open. There are no pieces on it.",
    "In this position, the {start_tok}{end_tok} is an open diagonal. It contains no pieces."
]


def sample_one(board: BoardRepr, frequency: dict, rng: random.Random) -> dict:
    # Length-proportional pick over the 26 diagonals. Diagonals span 2..8
    # squares; a 2-square corner carries far less information than the 8-square
    # main, so weighting by length gives every square an equal chance of
    # being in the queried diagonal (modulo the two diagonals per square).
    # No answer-side balance term. The frequency dict is still updated (via
    # answer_class) for cross-task coupling.
    diagonals = board.diagonals()
    diag_sqs  = rng.choices(diagonals, weights=[len(d) for d in diagonals], k=1)[0]
    f = line_facts(diag_sqs, board)

    use_cot = rng.random() < COT_RATIO

    if not f["ordered"]:
        fmt = {"start_tok": f["start_tok"], "end_tok": f["end_tok"]}
        a_t = DIAGONAL_EMPTY_ANSWERS_TOK
    elif use_cot:
        items = [
            (board.sq_tok(sq),
             None if board.piece_at(sq) == EMPTY_TOKEN else board.piece_at(sq))
            for sq in diag_sqs
        ]
        fmt = {
            "start_tok":        f["start_tok"],
            "end_tok":          f["end_tok"],
            "square_breakdown": format_square_breakdown(items),
        }
        a_t = DIAGONAL_COT_PRESENT_ANSWERS_TOK
    else:
        fmt = {
            "start_tok":    f["start_tok"],
            "end_tok":      f["end_tok"],
            "piece_counts": format_piece_counts(f["ordered"]),
        }
        a_t = DIAGONAL_DIRECT_PRESENT_ANSWERS_TOK
    q = rng.choice(DIAGONAL_QUESTIONS_TOK).format(**fmt)
    a = rng.choice(a_t).format(**fmt)
    return {
        "question":      q,
        "answer":        f"{a}\n\n{f['parse_tag']}",
        "question_type": NAME,
        "answer_class":  f["answer_class"],
    }
