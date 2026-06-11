"""Pure string-formatting helpers for natural-language prose in QA answers.

No knowledge of BoardRepr / chess.Board / FEN — callers resolve square and
piece tokens upstream (via BoardRepr.sq_tok / .piece_at) and pass plain
strings here. Keeps the abs/POV branch contained in BoardRepr.

Sources:
- `prose_list`, `format_piece_counts`, `format_square_breakdown` ported from
  the old `utils/build_qa_dataset.py` (now superseded by `datagen/`).
- `plural`, `join_oxford`, `line_token` ported from
  `depr/ab_chesslm/src/data/curriculum/tasks/_common.py`. The three
  range-token helpers (`file_range_token`, `rank_range_token`,
  `diagonal_token`) collapse into one `line_token` since callers always
  have pre-rendered start/end square tokens.

Excluded on purpose: curriculum's `line_facts`, `line_templates`,
`line_make_sample` (cross-task orchestration — violates the per-task-
isolation rule in plans/merge_data_pipelines.md), and `build_diagonals`
(geometric — lives on `BoardRepr.diagonals()`).
"""
from typing import Optional


# ---------------------------------------------------------------------------
# List joiners
# ---------------------------------------------------------------------------

def prose_list(items: list[str]) -> str:
    """'a' / 'a, and b' / 'a, b, and c'. Inherited from old build_qa_dataset.py.

    Note: slightly uglier than `join_oxford` for n=2 ('a, and b' vs 'a and b').
    Both are kept; pick per template phrasing taste.
    """
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


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


def line_token(start_tok: str, end_tok: str) -> str:
    """'<SQUARE_A1>-<SQUARE_A8>' range string for group-task prompts.

    Unifies the file/rank/diagonal range-token trio from curriculum's
    _common.py — since BoardRepr.sq_tok already handles abs vs POV
    rendering, all three reduce to the same '{start}-{end}' join.
    """
    return f"{start_tok}-{end_tok}"


# ---------------------------------------------------------------------------
# Group-task prose (file / rank / diagonal)
# ---------------------------------------------------------------------------

# TODO(nl-prose): "{count} {piece_tok}" treats the token as a noun and skips
# pluralization. Lift into a prompt_utils.py template family with phrasing
# variants once the merge is settled.
def format_piece_counts(counts_in_order: list[tuple[str, int]]) -> str:
    """[('<PIECE_WB>', 2), ('<PIECE_BR>', 1)] -> '2 <PIECE_WB>, and 1 <PIECE_BR>'."""
    return prose_list([f"{c} {tok}" for tok, c in counts_in_order])


# TODO(nl-prose): per-square phrasing is hand-rolled and not template-driven.
# Lift into prompt_utils.py with variants so CoT prose has phrasing diversity.
def format_square_breakdown(items: list[tuple[str, Optional[str]]]) -> str:
    """Per-square CoT walk.

    Each item is (sq_tok, piece_tok_or_None). None = empty square.
    e.g. [('<SQUARE_A1>', '<PIECE_WK>'), ('<SQUARE_A2>', None)]
         -> '<SQUARE_A1> has <PIECE_WK>, and <SQUARE_A2> is empty'
    """
    parts = [
        f"{sq_tok} is empty" if piece_tok is None else f"{sq_tok} has {piece_tok}"
        for sq_tok, piece_tok in items
    ]
    return prose_list(parts)
