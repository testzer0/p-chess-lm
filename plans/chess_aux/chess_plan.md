# Chess Endgame Project

## Overall Goals

Train a model that is:
 - able to play chess at a high level (>50% winrate against strong API models such as Gemini)
 - able to verbalize its reasoning and describe its plans in natural language

We follow an encoder-decoder framework:
 - **Encoder**: frozen LC0 network whose hidden states encode rich, position-aware board representations
 - **Decoder**: pretrained SmolLM3 3B connected to the encoder via a learned bridge

The key constraint: the decoder must learn to interpret encoder hidden states *without* access to LC0's move-prediction head output.

## Scope: King and Pawn Endgames

Starting with KP endgames, expanding complexity incrementally.

---

## Architecture

**Encoder (LC0 BT5, frozen)** — `chesslm/encoder/lc0_hf_bt5/`

Hidden states: `(batch, 64, 1024)`, 16 layers (embedding block + 15 transformer layers).

Canonicalization: LC0 encodes POV-relative. Default (v2/v2.1): un-flip black positions with `hidden[sq] = lc0_output[sq ^ 56]` so index 0 always = a1 (board-absolute). v3 skips the flip — hidden state index `i` stays as LC0's native "current player's square i", matching the POV-relative `<SQUARE_N>` token convention.

**Decoder (SmolLM3 3B, frozen)** — `hidden_size=2048`, 36 layers, 16 Q heads / 4 KV heads, `head_dim=128`.

### Bridge — Experiment A: Flamingo-style (current)

16 trainable `DenseXAttn` sublayers inserted before decoder layers 0, 2, 4, ..., 30.
x-attn sublayer `i` attends to encoder layer `i` (1-to-1 pairing, layers 0–15).
~470M trainable params. Full spec: `plans/chess_aux/kv_bridge.md`.

### Bridge — Experiment B: Dense Connector + LoRA (planned, start once A is training)

Multi-layer encoder features (layers 4, 8, 12, 15) concatenated channel-wise → 2-layer MLP projector → 64 context tokens prepended to decoder input. LoRA on decoder (~50–100M trainable). Motivated by 2024–2025 field consensus (InternVL, Qwen2-VL, LLaVA-OneVision all converge here). Simpler to implement; useful comparison against Experiment A.

---

## Training

### Stage 1 — Midtraining: board interpretation

Train the bridge on QA pairs so the decoder learns to read board state from encoder hidden states.

**QA tasks:**
1. **Piece location** — "What piece is on e4?" → "There is a black knight on e4. `<PIECE_BN>`"
   Special tokens `<SQUARE_XY>` / `<PIECE_XY>` act as learned per-square probe directions via Q-K attention, analogous to the per-square probes that achieved 100% accuracy (see `plans/chess_aux/probe.md`).

2. **Lookahead** — Given a position and a sequence of moves, predict where the moved pieces end up.
   Builds toward genuine forward search; harder than static piece location.

### Stage 2 — SFT: verbalization

Prompt Gemini with LC0 analysis trees; train on (position, explanation) pairs.
Eval: tree similarity metrics (Quartet distance, RF distance).

### Stage 3 — RL: recover playing strength

 - RLHF with LLM-as-judge, rubric-based rewards for explanation faithfulness
 - RLVR using LC0 best move as a verifiable reward signal

---

## Codebase

**Package:** `chesslm/` (renamed from `chess/` to avoid shadowing python-chess)

**Encoder** (`chesslm/encoder/lc0_hf_bt5/hf_model.py`):
 - `Lc0Bt4HFModel.from_pretrained(path, local_files_only=True)`
 - `model(planes, output_hidden_states=True).all_hidden_states` → tuple of 16 × `(B, 64, 1024)`
 - `model.input_planes_from_fen(start_fen, moves)` → `Tensor[112, 64]`

**Model** (`chesslm/model.py`, `chesslm/utils/model_utils.py`):
 - `DenseXAttn`: Flamingo-style gated cross-attention (MHA, relu² FFN at 2× hidden dim, tanh gates init=0)
 - `FlamingoChessLM.from_pretrained(decoder_path, device=...)` — `model.device`, `model.trainable_parameters()`
 - Smoke tests: `chesslm/smoke_test_model.py`

**Data pipeline** (`chesslm/utils/`):

1. `sample_positions.py` — streams Lichess PGN → `chesslm/raw_data/positions_vN.jsonl`
   Each line: `[start_fen, move_list, end_fen]`. Samples only from ply ≥ 10 (both players have moved at least 5 times). LC0 history window: up to 7 moves before `end_fen`.

2. `create_sft_dataset.py` — reads positions JSONL → HuggingFace Arrow datasets
   Key args: `--output-dir` (passed explicitly, saves `train/` and `eval/` inside it), `--n-positions` (unique positions to process), `--questions-per-position` (default 2), `--static-square-frac` (default 0.75).
   Total train examples = `n_positions × questions_per_position`.
   Eval positions are taken first and excluded from train to prevent leakage.
   Writes `dataset_config.json` alongside the dataset; training reads this to set `pov` and `new_tok_in_query`.

   Encoding variants (controlled by flags, recorded in `dataset_config.json`):
   - `--no-new-tok-in-query` (v2 legacy): plain-text square/piece names in question and answer prose
   - `--new-tok-in-query` (default, v2.1): special tokens (`<SQUARE_A1>`, `<PIECE_WK>`) in question AND answer prose
   - `--pov` (v3): POV-relative tokens (`<SQUARE_1>`…`<SQUARE_64>`) throughout; no hidden-state flip at encode time

   Question/answer templates live in `chesslm/utils/prompt_utils.py` (12 template lists: 6 for static_square, 6 for static_piece; plain and `_TOK` variants).
   Plain format vars: `{square}`, `{color}`, `{piece}`, `{squares}` (prose list of square names).
   `_TOK` format vars: `{sq_tok}`, `{piece_tok}`, `{squares}` (prose list of token strings) — `{color}` and `{piece}` are absent from all `_TOK` templates.

Current datasets in `chesslm/datasets/`:
 - `v1/` — 1M examples from 500k positions; plain-text questions, answer_class format was broken (do not use)
 - `v2/` — 10M examples from 5M positions (positions_v2.jsonl); plain-text questions, correct answer_class
 - `v2.1/` — same positions as v2; special tokens in question AND answer prose (`new_tok_in_query=True`)
 - `v3/` — same positions as v2; POV-relative `<SQUARE_1>`…`<SQUARE_64>` tokens, no hidden-state flip

**Probes** (`chesslm/probe_exps/probe.py`): see `plans/chess_aux/probe.md`.

---

## Status

**Phase 0 — Scope & Setup: DONE**

**Phase 1 — Encoder setup: DONE**
 - LC0 BT5 weights loaded, canonicalization scheme implemented.

**Phase 2 — Midtraining (Stage 1): IN PROGRESS**
 - `FlamingoChessLM` (Experiment A) implemented and smoke-tested.
 - **Next**: QA data pipeline — piece location pairs first, then lookahead pairs.
 - Experiment B (Dense Connector + LoRA) to be implemented in parallel once A is training.

**Phase 3 — SFT (Stage 2): NOT STARTED**

**Phase 4 — RL (Stage 3): NOT STARTED**
