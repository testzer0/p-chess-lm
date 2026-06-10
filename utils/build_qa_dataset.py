"""Build one (question_type, encoding) Arrow dataset from a positions shard.

One invocation = one HF Arrow dataset written to
  {output_dir}/{question_type}/{abs|rel}/train/
plus a sibling dataset_config.json recording the generation settings.

Mixing across question types is a train-time concern handled by a dataloader
wrapper over multiple Arrow datasets — not done here.

Eval / val / test generation is out of scope for this file (will be a separate
exhaustive enumerator with end_fen dedup).
"""
import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import chess
from datasets import Dataset

from chesslm.utils.prompt_utils import (
    SQ_QUESTIONS_TOK, SQ_OCCUPIED_ANSWERS_TOK, SQ_EMPTY_ANSWERS_TOK,
    PC_QUESTIONS_TOK, PC_PRESENT_ANSWERS_TOK, PC_ABSENT_ANSWERS_TOK,
    FILE_QUESTIONS_TOK, FILE_DIRECT_PRESENT_ANSWERS_TOK,
    FILE_COT_PRESENT_ANSWERS_TOK, FILE_EMPTY_ANSWERS_TOK,
    RANK_QUESTIONS_TOK, RANK_DIRECT_PRESENT_ANSWERS_TOK,
    RANK_COT_PRESENT_ANSWERS_TOK, RANK_EMPTY_ANSWERS_TOK,
    DIAGONAL_QUESTIONS_TOK, DIAGONAL_DIRECT_PRESENT_ANSWERS_TOK,
    DIAGONAL_COT_PRESENT_ANSWERS_TOK, DIAGONAL_EMPTY_ANSWERS_TOK,
)
from chesslm.utils.utils import (
    EMPTY_TOKEN,
    PIECE_TOKENS,
    POV_SQUARE_TOKENS,
    SQUARE_TOKENS,
    _PIECE_TO_TOKEN,
)


# ---------------------------------------------------------------------------
# Geometric entity enumeration (board-absolute; POV rendering is downstream)
# ---------------------------------------------------------------------------

def _files() -> list[tuple[int, ...]]:
    """8 files, each (rank-1 → rank-8) for that file."""
    return [tuple(chess.square(f, r) for r in range(8)) for f in range(8)]


def _ranks() -> list[tuple[int, ...]]:
    """8 ranks, each (file-a → file-h) for that rank."""
    return [tuple(chess.square(f, r) for f in range(8)) for r in range(8)]


def _diagonals() -> list[tuple[int, ...]]:
    """All length≥2 diagonals across both axes.

    Rising diagonals (anti-diagonals, ascending file & rank): start = lower-
    left end, end = upper-right end.
    Falling diagonals: start = lower-right end (descending file, ascending
    rank), end = upper-left end. Matches the user's <SQUARE_G1>-<SQUARE_A7>
    convention.
    Length-1 corner diagonals (a8, h1, a1, h8 alone) are skipped.
    """
    # TODO(diagonals): currently enumerating BOTH axes → 26 diagonals total
    # (13 rising + 13 falling). Earlier design notes referenced "15 diagonal"
    # which would mean one axis only (15 incl. length-1 corners, 13 without).
    # Revisit whether we want one axis or both before locking in data
    # generation — the choice affects how the model learns "diagonal" geometry.
    diags = []
    # Rising: rank - file = c, c ∈ [-6, 6]  →  13 diagonals
    for c in range(-6, 7):
        sqs = tuple(chess.square(f, f + c) for f in range(8) if 0 <= f + c < 8)
        diags.append(sqs)
    # Falling: file + rank = c, c ∈ [1, 13]  →  13 diagonals
    # Iteration order over r=0..7 produces (high-file, low-rank) → (low-file, high-rank),
    # matching the lower-right → upper-left canonical.
    for c in range(1, 14):
        sqs = tuple(chess.square(c - r, r) for r in range(8) if 0 <= c - r < 8)
        diags.append(sqs)
    return diags


# ---------------------------------------------------------------------------
# BoardRepr — normalizes absolute / POV coordinate systems behind one API
# ---------------------------------------------------------------------------

class BoardRepr:
    """One per FEN. Hides the absolute/POV distinction from sampling code.

    Internal keys are always python-chess board square indices (0–63, a1=0).
    `sq_tok` renders them as either <SQUARE_XY> (absolute) or <SQUARE_N>
    (POV-relative, with POV flip for black-to-move).
    """

    def __init__(self, fen: str, pov: bool):
        self.pov = pov
        board = chess.Board(fen)
        self._is_black_pov = pov and (board.turn == chess.BLACK)

        self._sq_to_piece: dict[int, str] = {}
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            self._sq_to_piece[sq] = (
                _PIECE_TO_TOKEN[(piece.color, piece.piece_type)] if piece else EMPTY_TOKEN
            )

        self._piece_to_sqs: dict[str, list[int]] = {tok: [] for tok in PIECE_TOKENS}
        for sq, tok in self._sq_to_piece.items():
            if tok != EMPTY_TOKEN:
                self._piece_to_sqs[tok].append(sq)

        # Sort piece-square lists by display order so prose / parse_tag concat
        # is consistent within an encoding. POV mode: sort by POV index.
        if self.pov:
            key_fn = (lambda sq: sq ^ 56) if self._is_black_pov else (lambda sq: sq)
            for tok in PIECE_TOKENS:
                self._piece_to_sqs[tok].sort(key=key_fn)
        # else: insertion order (board sq 0..63) is already the canonical order.

    @classmethod
    def from_fen(cls, fen: str, pov: bool) -> "BoardRepr":
        return cls(fen, pov)

    def entities_of_kind(self, kind: str) -> list:
        if kind == "square":   return list(range(64))
        if kind == "piece":    return list(PIECE_TOKENS)
        if kind == "file":     return _files()
        if kind == "rank":     return _ranks()
        if kind == "diagonal": return _diagonals()
        raise ValueError(f"Unknown entity kind: {kind!r}")

    def piece_at(self, board_sq: int) -> str:
        return self._sq_to_piece[board_sq]

    def squares_with(self, piece_tok: str) -> list[int]:
        return self._piece_to_sqs[piece_tok]

    def sq_tok(self, board_sq: int) -> str:
        if self.pov:
            pov_idx = board_sq ^ 56 if self._is_black_pov else board_sq
            return POV_SQUARE_TOKENS[pov_idx]
        return SQUARE_TOKENS[board_sq]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

# TODO(nl-prose): _prose_list and all the _format_* helpers below produce raw
# natural-language strings inline rather than going through prompt_utils.py
# templates. This is fine for a first cut but means (a) phrasing variation is
# baked into one place per helper, (b) there's no centralized place to audit
# the prose the model sees, and (c) the format is brittle to template edits
# made later. Before any production run, lift these into prompt_utils.py
# templates and sanity-check the resulting prose end-to-end.
def _prose_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _load_positions(path: str):
    with open(path) as f:
        for line in f:
            start_fen, moves, end_fen = json.loads(line)
            yield start_fen, moves, end_fen


# ---------------------------------------------------------------------------
# Question-type registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuestionSpec:
    """Everything the unified sampler needs for one question type.

    entity_kind:      which slice of BoardRepr to sample from.
    answer_tokens_fn: (entity, board) -> list[tok]. Used only for inverse-
                      frequency weighting; None disables weighting and the
                      sampler falls back to rng.choice.
    build_qa_fn:      (entity, board) -> (q_tmpls, a_tmpls, fmt, parse_tag,
                      answer_class). Owns the occupied/empty branching that's
                      specific to this QT.
    """
    entity_kind: str
    answer_tokens_fn: Optional[Callable]
    build_qa_fn: Callable


# --- piece_on_square --------------------------------------------------------

def _pos_answer_tokens(sq_key, board: BoardRepr) -> list[str]:
    # Only the piece on the square varies with entity choice; the sq_tok itself
    # is a 1-1 function of the entity and doesn't help balance the answer class.
    return [board.piece_at(sq_key)]


def _pos_build_qa(sq_key, board: BoardRepr):
    piece_tok = board.piece_at(sq_key)
    sq_tok    = board.sq_tok(sq_key)
    if piece_tok == EMPTY_TOKEN:
        fmt = {"sq_tok": sq_tok}
        return (SQ_QUESTIONS_TOK, SQ_EMPTY_ANSWERS_TOK, fmt,
                f"{sq_tok}{piece_tok}", [sq_tok, piece_tok])
    fmt = {"sq_tok": sq_tok, "piece_tok": piece_tok}
    return (SQ_QUESTIONS_TOK, SQ_OCCUPIED_ANSWERS_TOK, fmt,
            f"{sq_tok}{piece_tok}", [sq_tok, piece_tok])


# --- square_of_piece --------------------------------------------------------

def _sop_answer_tokens(piece_tok, board: BoardRepr) -> list[str]:
    keys = board.squares_with(piece_tok)
    if not keys:
        return [EMPTY_TOKEN]
    return [board.sq_tok(k) for k in keys]


def _sop_build_qa(piece_tok, board: BoardRepr):
    keys = board.squares_with(piece_tok)
    if not keys:
        fmt = {"piece_tok": piece_tok}
        return (PC_QUESTIONS_TOK, PC_ABSENT_ANSWERS_TOK, fmt,
                f"{piece_tok}{EMPTY_TOKEN}", [piece_tok, EMPTY_TOKEN])
    sq_toks = [board.sq_tok(k) for k in keys]
    fmt = {"piece_tok": piece_tok, "squares": _prose_list(sq_toks)}
    return (PC_QUESTIONS_TOK, PC_PRESENT_ANSWERS_TOK, fmt,
            f"{piece_tok}{''.join(sq_toks)}", [piece_tok] + sq_toks)


# --- pieces_on_group (file / rank / diagonal × direct / cot) ---------------

def _group_answer_tokens(entity_squares, board: BoardRepr) -> list[str]:
    """Answer-side tokens for weighting: the multiset of pieces on the group.

    Start/end tokens are 1-1 with entity choice so they don't help balance the
    answer distribution; left out here. Empty lines all share answer = (EMPTY,)
    and rely on the weighter's multiplicity correction to avoid over-sampling.
    """
    pieces = [board.piece_at(sq) for sq in entity_squares
              if board.piece_at(sq) != EMPTY_TOKEN]
    if not pieces:
        return [EMPTY_TOKEN]
    # Sort by canonical PIECE_TOKENS order so the answer tuple is stable across
    # entities that hold the same multiset.
    return sorted(pieces, key=PIECE_TOKENS.index)


# TODO(nl-prose): "{count} {piece_tok}" is the only phrasing right now — e.g.
# "2 <PIECE_WB>, and 1 <PIECE_BR>" — which is grammatically odd (token treated
# as a noun, no plural). Should become a template family in prompt_utils.py
# with several variants and a sanity-checked rendering on a real sample.
def _format_piece_counts(counts_in_order: list[tuple[str, int]]) -> str:
    """e.g. [(<PIECE_WB>, 2), (<PIECE_BR>, 1)]  →  '2 <PIECE_WB>, and 1 <PIECE_BR>'.

    The piece token is left as a token (not pluralized) — the count is the
    cardinality.
    """
    parts = [f"{c} {tok}" for tok, c in counts_in_order]
    return _prose_list(parts)


# TODO(nl-prose): per-square phrasing ("{sq_tok} is empty" / "{sq_tok} has
# {piece_tok}") is hand-rolled and not run through prompt_utils.py. Lift into
# a template family with several variants so CoT prose has phrasing diversity
# and can be audited centrally. Same caveat applies as for _format_piece_counts.
def _format_square_breakdown(entity_squares, board: BoardRepr) -> str:
    """Square-by-square walk for CoT prose."""
    parts = []
    for sq in entity_squares:
        sq_tok = board.sq_tok(sq)
        p      = board.piece_at(sq)
        if p == EMPTY_TOKEN:
            parts.append(f"{sq_tok} is empty")
        else:
            parts.append(f"{sq_tok} has {p}")
    return _prose_list(parts)


def _make_group_build_qa(q_tmpls, a_present_tmpls, a_empty_tmpls, variant: str):
    """Factory returning a build_qa_fn for one (group_kind × variant)."""
    assert variant in ("direct", "cot")

    def build(entity_squares, board: BoardRepr):
        start_tok = board.sq_tok(entity_squares[0])
        end_tok   = board.sq_tok(entity_squares[-1])

        # Counts in canonical PIECE_TOKENS order
        counts: dict[str, int] = defaultdict(int)
        for sq in entity_squares:
            p = board.piece_at(sq)
            if p != EMPTY_TOKEN:
                counts[p] += 1
        ordered = [(p, counts[p]) for p in PIECE_TOKENS if counts[p] > 0]

        if not ordered:
            fmt = {"start_tok": start_tok, "end_tok": end_tok}
            return (q_tmpls, a_empty_tmpls, fmt,
                    EMPTY_TOKEN, [start_tok, end_tok, EMPTY_TOKEN])

        parse_tag    = "".join(p * c for p, c in ordered)
        # Pieces in answer_class are emitted with repetition so the frequency
        # tracker reflects per-token incidence faithfully.
        pieces_flat  = [p for p, c in ordered for _ in range(c)]
        answer_class = [start_tok, end_tok] + pieces_flat

        if variant == "direct":
            fmt = {
                "start_tok":    start_tok,
                "end_tok":      end_tok,
                "piece_counts": _format_piece_counts(ordered),
            }
        else:  # cot
            fmt = {
                "start_tok":        start_tok,
                "end_tok":          end_tok,
                "square_breakdown": _format_square_breakdown(entity_squares, board),
            }
        return (q_tmpls, a_present_tmpls, fmt, parse_tag, answer_class)

    return build


# --- registry ---------------------------------------------------------------

QUESTION_REGISTRY: dict[str, QuestionSpec] = {
    "piece_on_square": QuestionSpec(
        entity_kind="square",
        answer_tokens_fn=_pos_answer_tokens,
        build_qa_fn=_pos_build_qa,
    ),
    "square_of_piece": QuestionSpec(
        entity_kind="piece",
        answer_tokens_fn=_sop_answer_tokens,
        build_qa_fn=_sop_build_qa,
    ),
    "pieces_on_file_direct": QuestionSpec(
        entity_kind="file",
        answer_tokens_fn=_group_answer_tokens,
        build_qa_fn=_make_group_build_qa(
            FILE_QUESTIONS_TOK, FILE_DIRECT_PRESENT_ANSWERS_TOK,
            FILE_EMPTY_ANSWERS_TOK, "direct"),
    ),
    "pieces_on_file_cot": QuestionSpec(
        entity_kind="file",
        answer_tokens_fn=_group_answer_tokens,
        build_qa_fn=_make_group_build_qa(
            FILE_QUESTIONS_TOK, FILE_COT_PRESENT_ANSWERS_TOK,
            FILE_EMPTY_ANSWERS_TOK, "cot"),
    ),
    "pieces_on_rank_direct": QuestionSpec(
        entity_kind="rank",
        answer_tokens_fn=_group_answer_tokens,
        build_qa_fn=_make_group_build_qa(
            RANK_QUESTIONS_TOK, RANK_DIRECT_PRESENT_ANSWERS_TOK,
            RANK_EMPTY_ANSWERS_TOK, "direct"),
    ),
    "pieces_on_rank_cot": QuestionSpec(
        entity_kind="rank",
        answer_tokens_fn=_group_answer_tokens,
        build_qa_fn=_make_group_build_qa(
            RANK_QUESTIONS_TOK, RANK_COT_PRESENT_ANSWERS_TOK,
            RANK_EMPTY_ANSWERS_TOK, "cot"),
    ),
    "pieces_on_diagonal_direct": QuestionSpec(
        entity_kind="diagonal",
        answer_tokens_fn=_group_answer_tokens,
        build_qa_fn=_make_group_build_qa(
            DIAGONAL_QUESTIONS_TOK, DIAGONAL_DIRECT_PRESENT_ANSWERS_TOK,
            DIAGONAL_EMPTY_ANSWERS_TOK, "direct"),
    ),
    "pieces_on_diagonal_cot": QuestionSpec(
        entity_kind="diagonal",
        answer_tokens_fn=_group_answer_tokens,
        build_qa_fn=_make_group_build_qa(
            DIAGONAL_QUESTIONS_TOK, DIAGONAL_COT_PRESENT_ANSWERS_TOK,
            DIAGONAL_EMPTY_ANSWERS_TOK, "cot"),
    ),
}


# ---------------------------------------------------------------------------
# Unified inverse-frequency weighter
# ---------------------------------------------------------------------------

def _weight_entities(entities, board: BoardRepr,
                     answer_tokens_fn: Callable, frequency: dict) -> list[float]:
    """weight(e) = 1 / ((sum_freq_of_answer_tokens(e) + 1) * multiplicity(e))

    multiplicity(e) is the number of entities sharing e's answer-token tuple —
    i.e. the size of e's answer class. Tuples are just the hashable form of the
    answer token list; grouping entities by tuple equality == grouping by
    answer class. For question types with unique answers per entity (e.g.
    square_of_piece's present pieces) this resolves to 1 and the factor is a
    no-op; for types where many entities collapse to the same class (32 empty
    squares → <EMPTY>, several open files → <EMPTY>) it pulls them back so the
    *class* sees uniform expected sampling rate.
    """
    answer_tuples = [tuple(answer_tokens_fn(e, board)) for e in entities]
    mult: dict[tuple, int] = defaultdict(int)
    for a in answer_tuples:
        mult[a] += 1
    weights = []
    for a in answer_tuples:
        freq_sum = sum(frequency.get(t, 0) for t in a)
        weights.append(1.0 / ((freq_sum + 1) * mult[a]))
    return weights


# ---------------------------------------------------------------------------
# Per-position sampling
# ---------------------------------------------------------------------------

def sample_one(board: BoardRepr, spec: QuestionSpec, question_type: str,
               frequency: dict, rng: random.Random) -> dict:
    entities = board.entities_of_kind(spec.entity_kind)
    if spec.answer_tokens_fn is None:
        chosen = rng.choice(entities)
    else:
        weights = _weight_entities(entities, board, spec.answer_tokens_fn, frequency)
        chosen  = rng.choices(entities, weights=weights, k=1)[0]

    q_t, a_t, fmt, parse_tag, answer_class = spec.build_qa_fn(chosen, board)
    q = rng.choice(q_t).format(**fmt)
    a = rng.choice(a_t).format(**fmt)
    return {
        "question":      q,
        "answer":        f"{a}\n\n{parse_tag}",
        "question_type": question_type,
        "answer_class":  answer_class,
    }


# ---------------------------------------------------------------------------
# Train dataset generation
# ---------------------------------------------------------------------------

def generate_train_records(positions, question_type: str, pov: bool,
                           questions_per_position: int,
                           rng: random.Random) -> list[dict]:
    spec      = QUESTION_REGISTRY[question_type]
    frequency: dict[str, int] = defaultdict(int)
    records   = []
    seen      = set()

    for start_fen, moves, end_fen in positions:
        key = (start_fen, tuple(moves))
        if key in seen:
            continue
        seen.add(key)

        board = BoardRepr.from_fen(end_fen, pov)
        for _ in range(questions_per_position):
            sample = sample_one(board, spec, question_type, frequency, rng)
            for cls in sample["answer_class"]:
                frequency[cls] += 1
            sample["start_fen"] = start_fen
            sample["moves"]     = moves
            sample["fen"]       = end_fen
            records.append(sample)

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Build one (question_type, encoding) Arrow dataset.")
    parser.add_argument("--positions", required=True,
                        help="Path to positions JSONL (one [start_fen, moves, end_fen] per line)")
    parser.add_argument("--question-type", required=True, choices=list(QUESTION_REGISTRY),
                        help="Question type to generate")
    parser.add_argument("--pov", action=argparse.BooleanOptionalAction, default=False,
                        help="Use relative (POV) square tokens; default absolute")
    parser.add_argument("--questions-per-position", type=int, default=2)
    parser.add_argument("--n-positions", type=int, default=-1,
                        help="Cap on positions to consume (default: use all)")
    parser.add_argument("--output-dir", required=True,
                        help="Root output dir; dataset is written to "
                             "{output_dir}/{question_type}/{abs|rel}/train/")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    encoding = "rel" if args.pov else "abs"
    out_root = Path(args.output_dir) / args.question_type / encoding
    out_root.mkdir(parents=True, exist_ok=True)

    positions = _load_positions(args.positions)
    if args.n_positions > 0:
        positions = (p for i, p in enumerate(positions) if i < args.n_positions)

    print(f"Generating {args.question_type} / {encoding}  "
          f"(pov={args.pov}, qpp={args.questions_per_position})")
    records = generate_train_records(
        positions               = positions,
        question_type           = args.question_type,
        pov                     = args.pov,
        questions_per_position  = args.questions_per_position,
        rng                     = rng,
    )

    random.Random(args.seed).shuffle(records)

    dataset = Dataset.from_list(records)
    train_dir = out_root / "train"
    dataset.save_to_disk(str(train_dir))
    print(f"  saved {len(dataset)} examples → {train_dir}")

    cfg = {
        "pov":                    args.pov,
        "encoding":               encoding,
        "question_type":          args.question_type,
        "questions_per_position": args.questions_per_position,
        "n_examples":             len(dataset),
        "seed":                   args.seed,
        # Kept for compatibility with existing training_utils._load_dataset_config,
        # which reads `new_tok_in_query`; tokens are always in query now.
        "new_tok_in_query":       True,
    }
    with open(out_root / "dataset_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  wrote config → {out_root / 'dataset_config.json'}")


if __name__ == "__main__":
    main()
