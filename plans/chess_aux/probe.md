# Probe Experiments

## Goal

Verify that piece locations and attack maps are linearly decodable from raw LC0 hidden states — the diagnostic for the KV bridge shortcutting problem.

## Key Architectural Finding

LC0's embedding block (`ip_emb_pre_w`) is a dense `(64×emb_dense, 64×12)` cross-square projection — even layer 0 hidden states mix information across all squares. There is no layer at which hidden states are purely per-square.

## Data

`chesslm/data/positions.jsonl` — JSONL where each line is `[start_fen, move_list, end_fen]`. `start_fen` is at most 7 plies before `end_fen` (matching LC0's 8-slot history). Generated from `chesslm/data/standard_partial_2026_04.pgn` via `chesslm/data_gen/sample_positions.py`.

Val set: 100 positions, balanced — 50 white-to-move + 50 black-to-move.

## Probes (`chesslm/probe_exps/probe.py`)

- `PieceProbe`: `nn.Linear(1024, 13)`, shared across all 64 squares. 13 classes: 0=empty, 1–6=white pieces (pawn..king), 7–12=black pieces. Absolute colors (not POV-relative), consistent with canonicalized hidden states.
- `AttackProbe`: `nn.Linear(1024, 25)`, shared across all 64 squares. 25 joint classes encoding `(white_attackers, black_attackers)`, each capped at 4.
- Both support `--probe-type {piece,attack}`, `--run-name`, `--no-class-weights`, `--patience`.
- Output dir: `probe_outputs/{run_name}_steps{N}_bs{B}_lr{lr}/layer_{idx}/`.
- Inverse-frequency class weighting enabled by default; early stopping patience 5 evals.

**Launch:**
```
sbatch chesslm/scripts/train_piece_probe.sh   # array=0,3,7,11,15
sbatch chesslm/scripts/train_attack_probe.sh  # array=0,3,7,11,15
```

## Results (layers 0, 3, 7, 11, 15 — balanced runs)

**Piece probe (shared linear, `val_acc`):**

| layer | best val_acc |
|-------|-------------|
| 0     | 0.8467      |
| 3     | 0.8455      |
| 7     | 0.8366      |
| 11    | 0.8452      |
| 15    | 0.7750      |

- Accuracy flat across layers 0–11, drops at 15 (final layer shaped toward policy head).
- Inverse-frequency weighting improved kings by ~+30pp and queens by ~+27pp at layer 0.

**Attack probe (shared linear, `joint_acc` over 25 classes):**

| layer | best joint_acc |
|-------|---------------|
| 0     | 0.4719        |
| 3     | 0.5584        |
| 7     | 0.5530        |
| 11    | 0.5695        |
| 15    | 0.2270        |

- Marginal accuracies (~60%) much better than joint.
- Layer 15 collapses almost completely.

## Critical Finding

A collaborator trained a *separate* linear probe per square, concatenating all 16 layers (1024×16, 13), achieving **100% accuracy** on all squares. Piece identity is linearly decodable per square, but the decoding direction is square-specific — a single shared projection cannot capture all 64 directions simultaneously. The 84% shared-probe result is an artifact of the shared architecture.

## Conclusion

Encoder hidden states contain all the information needed. The KV bridge has the capacity to read piece locations — whether it learns to depends on the training signal (the shortcutting problem). Diagnostic goal complete.
