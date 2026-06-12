"""Task: square_of_piece — list squares holding a queried piece species.

Self-contained: entity enumeration + frequency-aware weighter + build_qa all
inline. Per plans/merge_data_pipelines.md, no shared TaskSpec/weighter.
"""
import random
from collections import defaultdict

from utils.board_representation import BoardRepr
from datagen.prose import join_oxford
from utils.utils import EMPTY_TOKEN

NAME = "square_of_piece"

PC_QUESTIONS_TOK = [
    "What square(s) is {piece_tok} on?",
    "Where is {piece_tok} in this position?",
    "Locate {piece_tok} on the board.",
]

PC_PRESENT_ANSWERS_TOK = [
    "{piece_tok} can be found on {squares}.",
    "In this position, {piece_tok} is on {squares}.",
    "{piece_tok} occupies {squares}.",
]

PC_ABSENT_ANSWERS_TOK = [
    "There is no {piece_tok} in this position.",
    "This position has no {piece_tok}.",
]


def sample_one(board: BoardRepr, frequency: dict, rng: random.Random) -> dict:
    # Entities: the 12 piece tokens for this encoding (abs or POV). Answer-side
    # weighting tokens for a piece are the sq_toks where it's found (or
    # (EMPTY_TOKEN,) if absent). Multiplicity correction collapses the
    # (EMPTY,) group when several pieces are absent. The summed sq_tok freqs
    # naturally downweight high-cardinality species (pawns) and upweight rare
    # ones, so we deliberately omit an explicit piece_tok freq term — adding
    # one would force equal query rates across species despite very different
    # answer-information densities (1-square king vs 8-square pawn).
    entities = list(board.piece_tokens)
    answer_tuples = []
    for piece_tok in entities:
        board_sqs = board.squares_with(piece_tok)
        if board_sqs:
            answer_tuples.append(tuple(board.sq_tok(s) for s in board_sqs))
        else:
            answer_tuples.append((EMPTY_TOKEN,))

    mult: dict[tuple, int] = defaultdict(int)
    for a in answer_tuples:
        mult[a] += 1
    weights = [
        1.0 / ((sum(frequency.get(t, 0) for t in a) + 1) * mult[a])
        for a in answer_tuples
    ]
    piece_tok = rng.choices(entities, weights=weights, k=1)[0]

    board_sqs = board.squares_with(piece_tok)
    if not board_sqs:
        fmt = {"piece_tok": piece_tok}
        q = rng.choice(PC_QUESTIONS_TOK).format(**fmt)
        a = rng.choice(PC_ABSENT_ANSWERS_TOK).format(**fmt)
        parse_tag    = f"{piece_tok}{EMPTY_TOKEN}"
        answer_class = [piece_tok, EMPTY_TOKEN]
    else:
        sq_toks = [board.sq_tok(s) for s in board_sqs]
        # Randomize listing order so the model doesn't memorize a canonical
        # enumeration. Prose and parse_tag share the same shuffled order;
        # grader is multiset-match (chess_plan.md).
        rng.shuffle(sq_toks)
        fmt = {"piece_tok": piece_tok, "squares": join_oxford(sq_toks)}
        q = rng.choice(PC_QUESTIONS_TOK).format(**fmt)
        a = rng.choice(PC_PRESENT_ANSWERS_TOK).format(**fmt)
        parse_tag    = f"{piece_tok}{''.join(sq_toks)}"
        answer_class = [piece_tok] + sq_toks

    return {
        "question":      q,
        "answer":        f"{a}\n\n{parse_tag}",
        "question_type": NAME,
        "answer_class":  answer_class,
    }
