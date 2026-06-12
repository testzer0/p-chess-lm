"""Task: piece_on_file — "what pieces are on this file?"

Single task with two prose families (direct vs CoT) sampled per question.
File selection is uniform over the 8 files (no answer-side balance).
Compact piece-with-count encoding `<PIECE>?N` in parse_tag + answer_class
via `datagen.prose.encode_piece_counts`; reused across the line tasks and
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

NAME = "piece_on_file"

# Probability of routing a question through the CoT (per-square walk) template
# family vs the direct family. Hardcoded for now; lift to a CLI flag when
# build_qa_dataset.py gains task-level options.
COT_RATIO = 0.5

FILE_QUESTIONS_TOK = [
    "What piece(s) is on the file from {start_tok} to {end_tok}?",
    "Identify the piece(s) that occupy the {start_tok}{end_tok} file.",
    "In this position, the {start_tok}{end_tok} file contains what pieces(s)?"
]

FILE_DIRECT_PRESENT_ANSWERS_TOK = [
    "There are {piece_counts} on the file from {start_tok} to {end_tok}.",
    "The {start_tok}{end_tok} file contains {piece_counts}.",
    "In this position, {piece_counts} occupy the {start_tok}{end_tok} file."
]

FILE_COT_PRESENT_ANSWERS_TOK = [
    "Walking the file from {start_tok} to {end_tok}: {square_breakdown}.",
    "Going up the {start_tok}{end_tok} file: {square_breakdown}."
]

FILE_EMPTY_ANSWERS_TOK = [
    "There are no pieces on the file from {start_tok} to {end_tok}. It is an open file.",
    "The {start_tok}{end_tok} file is open. There are no pieces on it.",
    "In this position, the {start_tok}{end_tok} is an open file. It contains no pieces."
]


def sample_one(board: BoardRepr, frequency: dict, rng: random.Random) -> dict:
    # Uniform over the 8 files — no answer-side balance term. The frequency
    # dict is still updated (via answer_class) for cross-task coupling.
    file_sqs = rng.choice(board.files())
    f = line_facts(file_sqs, board)

    use_cot = rng.random() < COT_RATIO

    if not f["ordered"]:
        fmt = {"start_tok": f["start_tok"], "end_tok": f["end_tok"]}
        a_t = FILE_EMPTY_ANSWERS_TOK
    elif use_cot:
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
        "question_type": NAME,
        "answer_class":  f["answer_class"],
    }
