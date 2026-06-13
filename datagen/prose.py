"""String-formatting + line-task QA helpers shared across `datagen/tasks/*.py`.

The prose helpers (`join_oxford`, `plural`, `encode_piece_count`,
`format_piece_counts`, `format_square_breakdown`) are pure string ops and
have no knowledge of BoardRepr / chess.Board / FEN.

The line-task helpers (`line_piece_counts`, `line_facts`) take a tuple of
board squares + a BoardRepr and emit the shared (start_tok, end_tok,
ordered, parse_tag, answer_class) bundle used by piece_on_{file, rank,
diagonal}. They go through BoardRepr's API (`sq_tok`, `piece_at`,
`piece_tokens`) and contain no abs/POV branching themselves — that stays
in BoardRepr.

Sources:
- `format_piece_counts`, `format_square_breakdown` ported from the old
  `utils/build_qa_dataset.py` (now superseded by `datagen/`).
- `plural`, `join_oxford` ported from
  `depr/ab_chesslm/src/data/curriculum/tasks/_common.py`.
"""
from collections import defaultdict
from typing import Optional

from utils.board_representation import BoardRepr
from utils.utils import EMPTY_TOKEN


# ---------------------------------------------------------------------------
# List joiner
# ---------------------------------------------------------------------------

def join_oxford(items: list[str]) -> str:
    """Oxford-comma join: '' / 'a' / 'a and b' / 'a, b, and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


# ---------------------------------------------------------------------------
# Token formatting
# ---------------------------------------------------------------------------

def plural(token: str, n: int) -> str:
    """'<PIECE_WB>' -> '<PIECE_WB>s' for n != 1. Cosmetic only — never tokenized."""
    return f"{token}s" if n != 1 else token


def encode_piece_count(piece_tok: str, n: int) -> str:
    """Compact `<PIECE>?N` form for parse_tag + answer_class.

    n=1 -> bare token (e.g. '<PIECE_WK>'); n>1 -> token followed by the
    count (e.g. '<PIECE_WP>2'). Reused across piece_on_{file,rank,diagonal}
    and piece_count. Grader pairs with the regex
    `(<PIECE_[A-Z]+>|<SQUARE_[A-Z0-9]+>|<EMPTY>)(\\d+)?` to read counts back.
    """
    return f"{piece_tok}{n}" if n > 1 else piece_tok


# ---------------------------------------------------------------------------
# Group-task prose (file / rank / diagonal)
# ---------------------------------------------------------------------------

def format_piece_counts(counts_in_order: list[tuple[str, int]]) -> str:
    """[('<PIECE_WB>', 2), ('<PIECE_BR>', 1)] -> '2 <PIECE_WB>s and 1 <PIECE_BR>'.

    The piece token is pluralized via `plural()` for n != 1 (cosmetic prose
    only; parse_tag/answer_class use the compact count encoding, not this)."""
    return join_oxford([f"{c} {plural(tok, c)}" for tok, c in counts_in_order])


def format_square_breakdown(items: list[tuple[str, Optional[str]]]) -> str:
    """Per-square CoT walk.

    Each item is (sq_tok, piece_tok_or_None). None = empty square.
    e.g. [('<SQUARE_A1>', '<PIECE_WK>'), ('<SQUARE_A2>', None)]
         -> '<SQUARE_A1> has <PIECE_WK> and <SQUARE_A2> is empty'
    """
    parts = [
        f"{sq_tok} is empty" if piece_tok is None else f"{sq_tok} has {piece_tok}"
        for sq_tok, piece_tok in items
    ]
    return join_oxford(parts)


# ---------------------------------------------------------------------------
# Line-task QA structure (file / rank / diagonal)
# ---------------------------------------------------------------------------

def line_piece_counts(line_sqs: tuple, board: BoardRepr) -> list[tuple[str, int]]:
    """[(piece_tok, n), ...] in canonical piece_tokens order, non-zero only."""
    counts: dict[str, int] = defaultdict(int)
    for sq in line_sqs:
        p = board.piece_at(sq)
        if p != EMPTY_TOKEN:
            counts[p] += 1
    return [(p, counts[p]) for p in board.piece_tokens if counts[p] > 0]


def line_facts(line_sqs: tuple, board: BoardRepr) -> dict:
    """Shared start/end + parse_tag + answer_class bundle for line tasks.

    Returns:
        start_tok    : str  — sq_tok of the line's first square
        end_tok      : str  — sq_tok of the line's last square
        ordered      : list[(piece_tok, n)]  — empty for an open line
        parse_tag    : str  — compact "<PIECE>?N" concat, or "<EMPTY>"
        answer_class : list[str]  — [start_tok, end_tok, *pieces_flat] or
                                    [start_tok, end_tok, EMPTY_TOKEN]
    """
    start_tok = board.sq_tok(line_sqs[0])
    end_tok   = board.sq_tok(line_sqs[-1])
    ordered   = line_piece_counts(line_sqs, board)
    if not ordered:
        return {
            "start_tok":    start_tok,
            "end_tok":      end_tok,
            "ordered":      [],
            "parse_tag":    EMPTY_TOKEN,
            "answer_class": [start_tok, end_tok, EMPTY_TOKEN],
        }
    pieces_flat  = [encode_piece_count(p, c) for p, c in ordered]
    parse_tag    = "".join(pieces_flat)
    answer_class = [start_tok, end_tok] + pieces_flat
    return {
        "start_tok":    start_tok,
        "end_tok":      end_tok,
        "ordered":      ordered,
        "parse_tag":    parse_tag,
        "answer_class": answer_class,
    }
