"""Question and answer prose templates for SFT data generation.

Two variants per template group:
  plain  — uses human-readable square names / piece descriptions  (v2)
  _TOK   — uses special tokens as placeholders throughout          (v2.1, v3)

For _TOK variants the caller injects sq_tok / piece_tok / squares directly;
the same templates work for both board-absolute (v2.1) and POV-relative (v3)
since the token strings differ but the template structure is identical.

Static-square templates
-----------------------
Format vars (plain):   {square}, {color}, {piece}
Format vars (_TOK):    {sq_tok}, {piece_tok}            (occupied)
                       {sq_tok}                         (empty)

Static-piece templates
----------------------
Format vars (plain):   {color}, {piece}, {squares}      (squares = prose list of names)
Format vars (_TOK):    {piece_tok}, {squares}            (present; squares = prose list of tokens)
                       {piece_tok}                       (absent)
"""

# ---------------------------------------------------------------------------
# Static-square: plain
# ---------------------------------------------------------------------------

SQ_QUESTIONS: list[str] = [
    "What piece is on {square}?",
    "In this position, what piece occupies {square}?",
    "{square} contains what piece?",
]

SQ_OCCUPIED_ANSWERS: list[str] = [
    "There is a {color} {piece} on {square}.",
    "{square} contains a {color} {piece}.",
    "In this position, a {color} {piece} is on {square}.",
]

SQ_EMPTY_ANSWERS: list[str] = [
    "{square} is empty.",
    "There are no pieces on {square}.",
]

# ---------------------------------------------------------------------------
# Static-square: tokenized  (sq_tok replaces square name everywhere)
# ---------------------------------------------------------------------------

SQ_QUESTIONS_TOK: list[str] = [
    "What piece is on {sq_tok}?",
    "In this position, what piece occupies {sq_tok}?",
    "{sq_tok} contains what piece?",
]

SQ_OCCUPIED_ANSWERS_TOK: list[str] = [
    "There is a {piece_tok} on {sq_tok}.",
    "{sq_tok} contains a {piece_tok}.",
    "In this position, a {piece_tok} is on {sq_tok}.",
]

SQ_EMPTY_ANSWERS_TOK: list[str] = [
    "{sq_tok} is empty.",
    "There are no pieces on {sq_tok}.",
]

# ---------------------------------------------------------------------------
# Static-piece: plain
# ---------------------------------------------------------------------------

PC_QUESTIONS: list[str] = [
    "What square(s) are the {color} {piece}(s) on?",
    "Where are the {color} {piece}(s) in this position?",
    "The {color} {piece}(s) can be found on what squares?",
]

PC_PRESENT_ANSWERS: list[str] = [
    "The {color} {piece}(s) can be found on {squares}.",
    "In this position, the {color} {piece}(s) are on {squares}.",
    "The {color} {piece}(s) are at {squares}.",
]

PC_ABSENT_ANSWERS: list[str] = [
    "There are no {color} {piece}(s) in this position.",
    "This position has no {color} {piece}(s).",
]

# ---------------------------------------------------------------------------
# Static-piece: tokenized  (piece_tok in question; squares = prose of tokens)
# ---------------------------------------------------------------------------

PC_QUESTIONS_TOK: list[str] = [
    "What square(s) is {piece_tok} on?",
    "Where is {piece_tok} in this position?",
    "Locate {piece_tok} on the board.",
]

PC_PRESENT_ANSWERS_TOK: list[str] = [
    "{piece_tok} can be found on {squares}.",
    "In this position, {piece_tok} are on {squares}.",
    "{piece_tok} are at {squares}.",
]

PC_ABSENT_ANSWERS_TOK: list[str] = [
    "There is no {piece_tok} in this position.",
    "This position has no {piece_tok}.",
]


# ---------------------------------------------------------------------------
# Group queries (file / rank / diagonal): "what pieces are on this line?"
#
# TODO(nl-prose): DUMMY TEMPLATES — placeholders that wire up the
# build_qa_dataset.py generator end-to-end. Every list below has exactly one
# phrasing variant, which makes the resulting prose monotonous and unlikely to
# survive a sanity-check on real examples. Before production runs:
#   1. Expand each list to several variants (match the style density of
#      SQ_QUESTIONS_TOK / PC_QUESTIONS_TOK).
#   2. Render a handful of (qt, encoding) samples end-to-end and read them
#      out loud to catch grammatical oddities (the {piece_counts} prose in
#      particular reads stiffly — "2 <PIECE_WB>" with no pluralization).
#   3. Decide on the empty-line wording per kind ("open file" reads right;
#      "empty rank" / "empty diagonal" were guessed and may not be idiomatic).
#   4. Consider whether direct/cot variants should share or split Q templates
#      (currently shared) — the model needs *some* signal that distinguishes
#      direct from cot answers if we want them controllable at inference.
#
# A group is identified canonically by its endpoint pair, encoded as two square
# tokens ({start_tok}, {end_tok}). No new tokens are introduced for files /
# ranks / diagonals — the model reads the pair as the identifier.
#
# Two answer styles per group kind:
#   direct — bare count signature, e.g. "2 <PIECE_WP>, 1 <PIECE_BR> on …"
#   cot    — square-by-square walk, e.g. "<SQUARE_A1> has <PIECE_WR>, …"
#
# Format vars:
#   questions:        {start_tok}, {end_tok}
#   direct present:   {start_tok}, {end_tok}, {piece_counts}
#   cot    present:   {start_tok}, {end_tok}, {square_breakdown}
#   empty   (both):   {start_tok}, {end_tok}
# ---------------------------------------------------------------------------

# --- File ---  (DUMMY — 1 variant per list, see top-of-section TODO)

FILE_QUESTIONS_TOK: list[str] = [
    "What pieces are on the file from {start_tok} to {end_tok}?",
]

FILE_DIRECT_PRESENT_ANSWERS_TOK: list[str] = [
    "There are {piece_counts} on the file from {start_tok} to {end_tok}.",
]

FILE_COT_PRESENT_ANSWERS_TOK: list[str] = [
    "Walking the file from {start_tok} to {end_tok}: {square_breakdown}.",
]

FILE_EMPTY_ANSWERS_TOK: list[str] = [
    "There are no pieces on the file from {start_tok} to {end_tok}. It is an open file.",
]

# --- Rank ---  (DUMMY — 1 variant per list, see top-of-section TODO)

RANK_QUESTIONS_TOK: list[str] = [
    "What pieces are on the rank from {start_tok} to {end_tok}?",
]

RANK_DIRECT_PRESENT_ANSWERS_TOK: list[str] = [
    "There are {piece_counts} on the rank from {start_tok} to {end_tok}.",
]

RANK_COT_PRESENT_ANSWERS_TOK: list[str] = [
    "Walking the rank from {start_tok} to {end_tok}: {square_breakdown}.",
]

RANK_EMPTY_ANSWERS_TOK: list[str] = [
    "There are no pieces on the rank from {start_tok} to {end_tok}. It is an empty rank.",
]

# --- Diagonal ---  (DUMMY — 1 variant per list, see top-of-section TODO)

DIAGONAL_QUESTIONS_TOK: list[str] = [
    "What pieces are on the diagonal from {start_tok} to {end_tok}?",
]

DIAGONAL_DIRECT_PRESENT_ANSWERS_TOK: list[str] = [
    "There are {piece_counts} on the diagonal from {start_tok} to {end_tok}.",
]

DIAGONAL_COT_PRESENT_ANSWERS_TOK: list[str] = [
    "Walking the diagonal from {start_tok} to {end_tok}: {square_breakdown}.",
]

DIAGONAL_EMPTY_ANSWERS_TOK: list[str] = [
    "There are no pieces on the diagonal from {start_tok} to {end_tok}. It is an empty diagonal.",
]
