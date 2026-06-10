# Chess Endgame Project

## Overall Goals

Train a model that is:
- Able to play chess at a high level (>50% winrate against strong API models such as Gemini)
- Able to verbalize its reasoning and describe its plans in natural language

We follow an encoder-decoder framework:
- **Encoder**: frozen LC0 network whose hidden states encode rich, position-aware board representations
- **Decoder**: pretrained SmolLM3 3B connected to the encoder via a learned bridge

The key constraint: the decoder must learn to interpret encoder hidden states *without* access to LC0's move-prediction head output.

## Scope: King and Pawn Endgames

Starting with KP endgames, expanding complexity incrementally.

---

## Philosophy: Verbose Before Compressed

The LM should begin verbalizing explanations early, starting with very verbose atomic descriptions before distilling knowledge into more compact reasoning. For example, an explanation of checkmate might read:

> "If I play Qg7 in this position, there will be a white queen on g7 and a black king on g8. The black king can move to h8, h7, g7, f7, f8. However, all these squares are being attacked by at least one white piece. The white queen is also attacking the g8 square. Thus, black is in checkmate."

This mirrors how humans develop in proof-based settings — first each step is small and explicit, then patterns are recognized and compressed. Higher-level concepts (checkmate, forks, pins) emerge naturally from mastery of lower-level primitives.

---

## Roadmap: Four Foundational Skills

Before the model can verbalize complex reasoning, it must master four incremental skills:

### Skill 1 — Current Position Understanding
Conditioned on an encoder hidden state, the model can:
- Identify what piece is on a queried square
- Identify what square a queried piece is on
- Count how many white / black pieces are on the board

### Skill 2 — Current Position Movement / Attacks
Conditioned on an encoder hidden state, the model can:
- Identify how many attackers / defenders exist for a queried square
- List all squares a queried piece can move to or attack

### Skill 3 — Future Position Understanding
Conditioned on an encoder hidden state and a sequence of moves (the end position is **not** re-encoded by LC0), the model can:
- Identify what piece is on a queried square in the resulting position
- Identify what square a queried piece occupies after the move sequence
- Count piece totals in the resulting position

### Skill 4 — Future Position Movement / Attacks
Conditioned on an encoder hidden state and a sequence of moves, the model can:
- Identify how many attackers / defenders exist for a queried square in the resulting position
- List all squares a queried piece can move to or attack in the resulting position

After these four skills are solid, the roadmap extends to:
- **SFT (verbalization):** Prompt Gemini with LC0 analysis trees; train on (position, explanation) pairs. Eval via tree similarity metrics (Quartet distance, RF distance).
- **RL (recover playing strength):** RLHF with LLM-as-judge for explanation faithfulness; RLVR using LC0 best move as a verifiable reward signal.

---

## Architecture

**Encoder (LC0 BT5, frozen)** — `chesslm/encoder/lc0_hf_bt5/`

Hidden states: `(batch, 64, 1024)`, 16 layers (embedding block + 15 transformer layers).

Canonicalization: default (v2/v2.1) un-flips black positions via `hidden[sq] = lc0_output[sq ^ 56]` so index 0 = a1 (board-absolute). v3 skips the flip — index `i` stays as LC0's native POV-relative square `i`, matching `<SQUARE_N>` token convention.

**Decoder (SmolLM3 3B, frozen)** — `hidden_size=2048`, 36 layers, 16 Q heads / 4 KV heads, `head_dim=128`.

### Bridge Experiments

Three bridge architectures are under consideration. Initial testing on Skill 1 will compare expressiveness and sample efficiency across all three before committing to a primary approach. Full specs: `plans/chess_architectures.md`.

| | Experiment A (Flamingo) | Experiment B (LLaVA) | Experiment C (KV Proj) |
|---|---|---|---|
| Trainable params | ~470M | ~83M | ~612M |
| Decoder | fully frozen | frozen + LoRA | frozen + LoRA |
| Encoder info per layer | 1-to-1 x-attn pairing | all 16 concatenated | all 16 concatenated |
| Decoder sees encoder | via new x-attn sublayers | once, as prefix tokens | via KV cache at each layer |
| New modules | 16 DenseXAttn sublayers | 1 MLP + positional embeds | 1 LayerNorm + 36×2 linears |

---

## Data

### Existing Datasets (`chesslm/datasets/`)

| Version | Description |
|---------|-------------|
| v1 | 1M examples / 500k positions; plain-text questions; broken answer_class — **do not use** |
| v2 | 10M examples / 5M positions; plain-text questions; correct answer_class |
| v2.1 | Same positions as v2; special tokens (`<SQUARE_A1>`, `<PIECE_WK>`) in question and answer |
| v3 | Same positions as v2; POV-relative `<SQUARE_1>`–`<SQUARE_64>` tokens; no hidden-state flip |

These cover **Skill 1** (static piece location). New datasets are needed for Skills 2, 3, and 4.

### Data Pipeline

- `chesslm/utils/sample_positions.py` — streams Lichess PGN → `chesslm/raw_data/positions_vN.jsonl`
- `chesslm/utils/create_sft_dataset.py` — positions JSONL → HuggingFace Arrow datasets

Full pipeline details: `plans/chess_aux/train.md`.

---

## Codebase

**Package:** `chesslm/`

**Encoder** (`chesslm/encoder/lc0_hf_bt5/hf_model.py`):
- `Lc0Bt4HFModel.from_pretrained(path, local_files_only=True)`
- `model(planes, output_hidden_states=True).all_hidden_states` → tuple of 16 × `(B, 64, 1024)`
- `model.input_planes_from_fen(start_fen, moves)` → `Tensor[112, 64]`

**Models** (`chesslm/models/`):
- `base.py` — `ChessLM` Protocol + shared helpers: `init_new_token_embeddings`, `apply_lora`, `unwrap_decoder`, `decoder_trainable_params`, `save_decoder_state`, `load_decoder_state`
- `flamingo.py` — `FlamingoChessLM` + `DenseXAttn`. Constructor: `(decoder, n_new_tokens, lora_rank=-1, x_attn_kwargs)`. `lora_rank=-1` freezes decoder (default); `lora_rank=0` full training; `lora_rank>0` LoRA.
- `llava.py` — `LLaVAChessLM`. Constructor: `(decoder, n_new_tokens, lora_rank=0)`.
- `kv_proj.py` — `KVProjChessLM`. Constructor: `(decoder, n_new_tokens, proj_mode, lora_rank=0)`.
- All three expose: `from_pretrained(decoder_path, lora_rank=..., device=...)`, `model.device`, `model.trainable_parameters()`
- Smoke tests: `chesslm/smoke_tests/`

**Training** (`chesslm/train.py`): full spec in `plans/chess_aux/train.md`.

**Probes** (`chesslm/probe_exps/probe.py`): see `plans/chess_aux/probe.md`.

---

## Status

**Phase 0 — Scope & Setup: DONE**

**Phase 1 — Encoder setup: DONE**
- LC0 BT5 weights loaded, canonicalization scheme implemented.

**Phase 2 — Skill 1 (Current Position Understanding): IN PROGRESS**
- All three architectures implemented and smoke-tested.
- v2, v2.1, v3 datasets built for piece-location QA.
- Probe experiments complete: piece identity linearly decodable per square (100% with per-square probes); shared probe ~84% due to square-specific decoding directions. Attack probe joint acc ~57% (marginals ~60%), collapses at layer 15. See `plans/chess_aux/probe.md`.
- Exp 1 (architecture comparison, 16 runs): complete. LLaVA (Exp B) is the clear winner — reaches 100% on both question types; see `chesslm/experiments/exp1.md`.
- Exp 2 (Flamingo vs LLaVA × frozen vs LoRA decoder × 3 LRs, 30 runs): running. Key goals: disentangle bridge architecture from decoder trainability, find optimal LR, test Flamingo with fixed gate init (α=2, W_O=0); see `chesslm/experiments/exp2.md`.

**Phase 3 — Skill 2 (Movement / Attacks): NOT STARTED**

**Phase 4 — Skills 3 & 4 (Future Position): NOT STARTED**

**Phase 5 — SFT (verbalization): NOT STARTED**

**Phase 6 — RL (playing strength): NOT STARTED**

---

*Auxiliary docs: `plans/chess_architectures.md`, `plans/chess_data.md`, `plans/chess_aux/train.md`, `plans/chess_aux/probe.md`*
*Prior drafts: `plans/chess_aux/chess_plan.md`, `plans/chess_aux/chess_plan_updated.md`*
