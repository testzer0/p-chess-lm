"""Task: square_of_piece — list squares holding a queried piece species."""
import random
from collections import defaultdict

from utils.board_representation import BoardRepr
from datagen.prose import join_oxford
from utils.utils import EMPTY_TOKEN

NAME = "square_of_piece"
MAX_UNIQUE_QUERIES = 12

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


def _choose_entity(board: BoardRepr, frequency: dict, rng: random.Random,
                   exclude: set[str]) -> str:
    """Answer-side balance over `board.piece_tokens \\ exclude`.

    Summed sq_tok freqs naturally downweight high-cardinality species; mult
    correction collapses the (EMPTY,) group when several pieces are absent.
    """
    entities = [p for p in board.piece_tokens if p not in exclude]
    answer_tuples: list[tuple[str, ...]] = []
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
    return rng.choices(entities, weights=weights, k=1)[0]


def _render(piece_tok: str, board: BoardRepr, rng: random.Random) -> dict:
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
        # enumeration. Prose and parse_tag share the same shuffled order.
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


def sample_n(board: BoardRepr, frequency: dict, rng: random.Random, n: int) -> list[dict]:
    n = min(n, MAX_UNIQUE_QUERIES)
    seen: set[str] = set()
    out: list[dict] = []
    while len(out) < n:
        p = _choose_entity(board, frequency, rng, exclude=seen)
        seen.add(p)
        out.append(_render(p, board, rng))
    return out


# `sample_one` and `sample_all` are synthesized in `datagen/tasks/__init__.py`.
