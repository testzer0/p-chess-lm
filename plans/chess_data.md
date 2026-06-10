# Chess Data Spec

## Source

Lichess PGN games → `chesslm/utils/sample_positions.py` → `chesslm/raw_data/positions_vN.jsonl`

Each line: `[start_fen, move_list, end_fen]`
- `end_fen` is the position of interest
- `move_list` contains up to 7 moves before `end_fen` (LC0's history window)
- `start_fen` is the board state at the start of `move_list`
- Only positions sampled from ply ≥ 10 (both players have moved at least 5 times)

QA pairs are generated from positions JSONL via `chesslm/utils/create_sft_dataset.py` → HuggingFace Arrow datasets with `train/` and `eval/` splits. Eval positions are held out from train.

---

## Encoding Formats

Stage 1 sweeps confirmed tokens-in-prose dominates plaintext prose, so plaintext is **deprecated**. Going forward there are two encodings, distinguished by how square tokens are oriented:

### Absolute board encoding (`pov=False`)

Special tokens replace square and piece references everywhere.
- Square tokens: `<SQUARE_A1>` … `<SQUARE_H8>` (64 tokens, board-absolute)
- Piece tokens: `<PIECE_WP>`, `<PIECE_WN>`, …, `<PIECE_BK>`, `<EMPTY>` (13 tokens)
- `<SQUARE_A1>` always refers to a1 regardless of whose turn it is
- Encoder canonicalization: black positions un-flipped (`sq ^ 56`) so hidden state index 0 = a1
- Answer parse tag: same special token, already present in the answer prose

Example:
```
Q: What piece is on <SQUARE_E4>?
A: There is a <PIECE_WN> on <SQUARE_E4>. <PIECE_WN>
```

### Relative (POV) board encoding (`pov=True`)

Square tokens are oriented to the side-to-move.
- Square tokens: `<SQUARE_1>` … `<SQUARE_64>` (64 tokens, POV-relative)
  - `<SQUARE_1>` = current player's a1 (always the near-left corner)
  - `<SQUARE_64>` = current player's h8
- Piece tokens: same absolute color tokens (`<PIECE_WN>` etc.)
- Encoder canonicalization: no hidden state flip — LC0's native ordering, so hidden state index `i` matches `<SQUARE_i>`
- Answer parse tag: same as above

Example (white to move):
```
Q: What piece is on <SQUARE_29>?   (= e4 from white's POV)
A: There is a <PIECE_WN> on <SQUARE_29>. <PIECE_WN>
```

---

## Existing Datasets (`chesslm/datasets/`)

| Version | Encoding | Positions | Examples | Notes |
|---------|----------|-----------|----------|-------|
| v1 | plaintext (deprecated) | 500k | 1M | Broken answer_class — **do not use** |
| v2 | plaintext (deprecated) | 5M | 10M | Plain-text prose; superseded by v2.1 |
| v2.1 | absolute | 5M (same as v2) | 10M | Stage 1 question types |
| v3 | relative (POV) | 5M (same as v2) | 10M | Stage 1 question types |

All existing datasets cover **Stage 1 question types only**. New dataset versions will continue to come in absolute/relative pairs (one of each per dataset bump).

---

## Question Types by Stage

Question types are described here at the semantic level. Templates live in `chesslm/utils/prompt_utils.py`; sampling/dispatch logic lives in `chesslm/utils/generate_sft_data.py`.

### Stage 1 — Current Position Understanding

All questions are conditioned on the encoder hidden state of `end_fen` only. No move sequences.

1. **Piece on square**: Given a square, identify what piece (or empty) occupies it.
2. **Square of piece**: Given a piece type and color, identify what squares it occupies.
3. **Piece count**: Given a color, count how many pieces of that color are on the board. Side-balanced (asks decomposition for both colors); FEN distribution controlled at sampling time so no in-function frequency weighting is applied.
4. **Material count**: Given a color, sum the material value of that color's pieces. Same FEN-controlled distribution as piece count.
5. *(future Stage 1 additions: rank/file/diagonal queries — "what pieces are on rank 4?", "which squares on the long diagonal hold black pieces?")*

### Stage 2 — Current Position Movement / Attacks

All questions conditioned on `end_fen` encoder hidden state only.

6. **Legal moves for piece**: Given a square, list all squares the piece on that square can legally move to.
7. **Attackers of square**: Given a square, identify how many white / black pieces are attacking it.
8. **Defenders of square**: Given a square, identify how many white / black pieces are defending it.

### Stage 3 — Future Position Understanding

Questions conditioned on the encoder hidden state of `start_fen` + `move_list` (the end position is **not** re-encoded). The model must mentally apply the move sequence.

9. **Piece on square (future)** / **Square of piece (future)** / **Piece count (future)** / **Material count (future)**.

### Stage 4 — Future Position Movement / Attacks

Conditioned on `start_fen` encoder + `move_list`, applied mentally. Legal moves, attackers, defenders — same as Stage 2 but in the resulting position.

---

## Answer Format

Every answer ends with a structured parse tag that is machine-checkable. The answer body is always natural language prose; the parse tag is appended at the end as a structured output for eval.

| Question type | Parse tag |
|---|---|
| Piece on square | `<PIECE_XY>` or `<EMPTY>` |
| Square of piece | `<SQUARE_XY>` (absolute) or `<SQUARE_N>` (relative) |
| Piece count / material count | `<COUNT_N>` for piece count (0–16); material is the sum of piece values — token format TBD |
| Legal moves | space-separated destination square tokens, e.g. `<SQUARE_D2> <SQUARE_F2> …`; `<EMPTY>` if the source square has no piece |
| Attackers / defenders | `<COUNT_N>` |

Example (legal moves):
```
Q: What squares can the piece on <SQUARE_E4> move to?
A: The <PIECE_WN> on <SQUARE_E4> can move to <SQUARE_D2>, <SQUARE_F2>, <SQUARE_D6>, <SQUARE_F6>, <SQUARE_C3>, <SQUARE_G3>, <SQUARE_C5>, and <SQUARE_G5>.
   <SQUARE_D2> <SQUARE_F2> <SQUARE_D6> <SQUARE_F6> <SQUARE_C3> <SQUARE_G3> <SQUARE_C5> <SQUARE_G5>

Q: What squares can the piece on <SQUARE_E4> move to?
A: There is no piece on <SQUARE_E4>. <EMPTY>
```

`<COUNT_N>` tokens (N = 0–16) and the move-list square token format are not yet in the tokenizer — the token set must be extended when Stage 2–4 datasets are built.

---

## Generation Architecture

The two existing files — `chesslm/utils/generate_sft_data.py` (per-question-type sampling helpers) and `chesslm/utils/create_sft_dataset.py` (mix-ratio orchestration writing one combined dataset) — are being merged into a single file that produces **one dataset per (question_type, encoding) pair**. Mix ratios across question types move out of generation and into the dataloader at train time.

### Unified entry point

Single file, single invocation = single Arrow dataset on disk.

```
python -m chesslm.utils.build_qa_dataset \
    --positions          <positions.jsonl>     # SFEN, MVLIST, EFEN list
    --question-type      <qt>                  # one of the registered types
    --pov                {true,false}          # absolute vs relative encoding
    --questions-per-pos  <M>                   # train mode only
    --output-dir         <path>
```

Output path: `chesslm/datasets/{question_type}/{abs|rel}/{train,val,test}/`.

The generator is **stateless about splits**: each invocation consumes one positions shard and writes one split (train, val, or test) of one (encoding, question_type). Train/val/test partitioning of positions is done upstream — currently out of scope for this file.

### Generation order (train pipeline, scope of this file)

For now the generator handles **train-mode only**:

- Take a positions shard, generate `M` weighted-sampled questions per position for one (qt, encoding).
- Output: an Arrow dataset + a `dataset_config.json` recording `{pov, question_type, n_positions, questions_per_position}`.

Eval/val/test generation is a separate concern and will be tackled later. Planned flow (not implemented yet): generate **eval first** on a held-out positions shard (exhaustively enumerating every entity per position — 64 piece-on-square, 12 square-of-piece, 8 rank, 8 file, 15 diagonal, etc.), record the set of `end_fen`s, then run train generation on a separate shard with a dedup check against that set. Val and test get the same exhaustive treatment on their own shards.

### Three abstractions inside the file

**1. `BoardRepr`** — a per-FEN object that normalizes absolute vs relative coordinate systems behind one interface:
- `entities_of_kind(kind) → list[entity]`, where `kind ∈ {square, piece, rank, file, diagonal, …}`
- `key_to_piece(entity) → piece_tok` (for square-style entities)
- `squares_of(entity) → list[sq_tok]` (for piece-style or geometric entities)
- `sq_tok(entity) → str` (returns absolute or POV token depending on construction)

Built once per FEN via `BoardRepr.from_fen(fen, pov)`. Eliminates the four parallel `_static_*` / `_static_*_pov` helpers.

**2. Unified inverse-frequency weighter**

A single function balances each dataset's answer-token distribution. Per question_type metadata in the registry:
- `answer_tokens_fn(entity, board) → list[tok]` — which tokens count as the answer for weighting purposes (the input-side token is excluded because it doesn't differentiate entities of the same kind)

Weighting rule: `weight(e) = 1 / ((sum_of_freq_of_answer_tokens(e) + 1) × multiplicity(e))`, where `multiplicity(e)` is the number of entities sharing the same answer-token set (auto-computed by grouping entities by their `answer_tokens_fn` output). Multiplicity correction reduces to `1` for question types where each entity has a unique answer (e.g. `square_of_piece`), and gives non-trivial normalization where it's needed (e.g. `piece_on_square` — 32 empty squares all share answer = `<EMPTY>`, so dividing by `class_counts[<EMPTY>]=32` keeps the empty class from dominating).

The frequency dict is scoped to one invocation = one question type, so each QT's answer space is independently balanced. No cross-QT contamination.

Question types where the FEN distribution is controlled upstream (piece count, material count) declare `weighting=None` and the sampler short-circuits — no entity choice, just compute the answer.

**3. Question-type registry**

```python
QUESTION_REGISTRY[question_type] = {
    "entity_kind":       "square" | "piece" | "rank" | "file" | "diagonal" | None,
    "answer_tokens_fn":  callable(entity, board) → list[tok]      # None if weighting=None
    "templates":         (q_tmpls, present_tmpls, absent_tmpls)   # or similar shape
    "parse_tag_fn":      callable(entity, board) → str            # structured tail
}
```

Adding a new question type becomes: (a) add templates to `prompt_utils.py`, (b) optionally add a new entity kind to `BoardRepr` if a geometric primitive is new, (c) one row in the registry. No per-type sampling code.

### Train-time mixing (separate concern)

Datasets are saved as Arrow, loaded at train time as `datasets.IterableDataset`, and combined by a wrapper that yields from N iterables according to a fixed weight vector. The wrapper, mix-ratio config format, and dataloader integration are TBD and not part of `build_qa_dataset.py`.

---

## Pipeline Stages & Distribution Control

End-to-end the data pipeline has three independently tunable weighting layers:

```
.jsonl shards of games
   ↓  [optionally weighted] position sampling          ← `sample_positions.py`
positions JSONL shard
   ↓  [optionally weighted] QA generation               ← `build_qa_dataset.py`
per-QT Arrow datasets
   ↓  [optionally weighted] dataloader mix              ← TBD wrapper
training stream
```

Each stage targets a different axis of imbalance, and the choice of where to fix something matters because **no single stage can fix everything**.

### What each stage can and can't balance

**Position sampling** controls the dataset's raw piece-on-board distribution. Position-aware sampling here is the only knob that can address structural piece rarity — e.g. raising queen exposure by oversampling positions that have queens, or raising `<EMPTY>` exposure by oversampling endgame positions with open files. Nothing downstream can manufacture queens that aren't on the board.

**QA generation** (the inverse-frequency weighter in `build_qa_dataset.py`) controls which **answer classes** get sampled per position. It balances at the answer-class level — given a fixed position, every unique answer multiset gets roughly equal sampling probability. It can't fix imbalances that come from the position distribution itself.

**Dataloader mixing** controls cross-QT exposure. If a token is well-balanced in one QT but skewed in another, mix weights can up-weight the balanced QT to compensate. E.g. `piece_on_square` balances all 13 piece classes (12 pieces + `<EMPTY>`) per mult correction, so the mix can lean on it for piece-token coverage even if `pieces_on_file_*` is pawn-heavy.

### The per-token skew, documented

Audit on `pieces_on_file_direct` (5000 positions × 1 question each, abs encoding) shows the structural skew clearly:

| token | count | notes |
|-------|------:|-------|
| `<PIECE_BP>` / `<PIECE_WP>` | ~2690 each | pawns appear in ~75% of files |
| `<PIECE_WR>` / `<PIECE_BR>` | ~900 each | rooks in ~25% of files |
| `<PIECE_WK>` / `<PIECE_BK>` | ~580 each | kings on 1 file per side per position |
| `<PIECE_WQ>` / `<PIECE_BQ>` | ~510 each | queens often off the board |
| `<EMPTY>` | ~480 | open files in ~10% of file samples |

The pawn-token dominance is **not a weighter bug**:

1. *Most files contain at least one pawn*, often two (one per color). Each non-empty file contributes 2–3 piece tokens to per-token counts; each empty file contributes one `<EMPTY>`.
2. *Mult correction caps empty-class sampling* — even with 3/8 open files in a position, the empty class collectively gets the weight of one slot, by design (it balances per-class, not per-token).
3. *No weighter inside `build_qa_dataset` can fix this*, even without mult: pawns co-occur with every other piece class, so sampling any non-empty file feeds `freq[pawn]`. Class-balanced sampling produces token-skewed output whenever some tokens appear in many classes — which is exactly the case for pawns.

The same logic applies to `<SQUARE_H1>`-style under-representation in `square_of_piece` (kings rarely sit there) and to queen under-representation in any pieces-on-line QT.

### Why this likely doesn't matter

Per-token *uniformity* is not actually what we need — what we need is *enough* exposure per token for the embedding to learn it. 510 queen-token occurrences in 5000 records is an order of magnitude above the threshold where SmolLM3 would fail to learn `<PIECE_WQ>`. The pawn over-exposure doesn't degrade queen learning; embedding updates for `<PIECE_WQ>` are independent of how often `<PIECE_BP>` gets updated.

**Consequence for the smoke test**: the right invariant to check is "every expected answer class hits a minimum count," not "max/min ratio under some threshold." A class-starvation check — flagging any token that appears in fewer than ~N/(n_classes × k) records — is what we should test. The current ratio-based test (`chesslm/smoke_tests/test_qa_distribution.py`) overstates the problem.

### When to escalate to upstream stages

If a token's exposure drops below the "enough to learn" threshold in the training mix, address it at the appropriate stage:

- **Token rarity from board reality** (queens, corner squares): bias position sampling toward positions containing those tokens. `sample_positions.py` can take per-piece or per-square target counts.
- **Cross-QT imbalance** (one QT's natural distribution differs from another's): tune dataloader mix weights to up-weight the QT with better coverage of the under-exposed token.
- **Answer-class starvation within a QT** (`(<PIECE_WQ>, <PIECE_WQ>)` class essentially never sampled because two queens never coexist on a file): accept it — the class is rare in reality and the model doesn't need to learn it as a primary pattern.

No work planned on the upstream knobs until a downstream symptom (eval miss on a particular token kind) actually motivates it.

---

## Datasets To Build

| Stage | Status |
|---|---|
| Stage 1 (piece-on-square, square-of-piece) | Done (v2.1 abs, v3 rel) — pre-refactor combined format |
| Stage 1 — re-built per-QT under new generator | Not started |
| Stage 1 extensions (piece count, material count, rank/file/diagonal) | Not started — built on top of the unified generator |
| Stage 2 (legal moves, attackers, defenders) | Not started |
| Stage 3 (future-position counterparts) | Not started |
| Stage 4 (future-position movement / attacks) | Not started |
