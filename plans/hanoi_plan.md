## Chess Project

### Overall Goals

The goal of this project is to train a model that is
 - able to play chess at a high level ( >50% winrate against some strong API model such as gemini-3.1 )
 - able to verbalize its reasoning and describe its plans

 There is not that much literature on training language models how to play chess. To start with an easier
 task, we will assume access to a strong chess-only model (transformer architecture) called LC0 (you can read more here: https://lczero.org/blog/2024/02/transformer-progress/)

 The plan is to train an encoder-decoder model with the encoder fixed as this LC0 network, and the idea is that we know that the
 hidden states of the LC0 network are meaningful weights with respect to chess playing ability. Then, the goal is to train a language model
(likely from a checkpoint of an open source model) to be able to interpret the hidden states without any signal from the gold predictions 
from the LC0 head (that would predict the next move at an extremely high level). The goal is not to post-hoc explain LC0's choice, but rather for the 
language model to understand and develop its own reasoning for the best move at each position.

Before starting with chess, let's investigate a toy setting, namely Tower of Hanoi. We will mimic this encoder-decoder
framework, so the first task is to train an encoder (llm) from scratch that always takes in a sequence of 3 tokens and has a linear head with the 6 legal moves (disk on spindle 1 -> 2,3; disk on spindle 2 -> 1,3; disk on spindle 3 -> 1, 2).
It will also have another head that predicts the number of moves to the final termination state. We will form the embeddings as linear projections as described in the linked article above.
These are all programatically computable as there exists a fixed algorithm to solve tower of hanoi with k disks.

We will assume a maximum of k=8 discs, but there can be fewer than 8 in the examples that we generate.

### Encoder-Decoder Bridge (Tower of Hanoi → Language)

Once the encoder is trained, the next step is to connect it to a language model decoder (planned: SmolLM) so the
composite model can answer natural language questions about a position (e.g. "what is the best next move?",
"which disks are on peg 3?").

**Architecture:**

The encoder produces last-layer hidden states of shape `(3, d_model)` — one vector per peg token. We inject these
into the decoder as a prepended KV cache so the decoder can attend to the position representation at every layer
without consuming any of the decoder's context tokens.

Concretely, a learned `KVProjector` module maps each encoder hidden state to a key and value vector for every
layer of the decoder:

```
encoder last-layer hidden states: (3, d_enc)
        ↓  KVProjector (linear, per decoder layer)
injected KV cache: num_decoder_layers × (3, d_dec_head)  [keys and values separately]
```

The KV cache is prepended to the decoder's self-attention at each layer. The decoder then processes a
natural language question conditioned on this context and generates a free-form answer.

**Training:**

- The encoder is frozen after supervised pretraining on optimal move prediction.
- Only the `KVProjector` and the decoder (fine-tuned from SmolLM-135M-Instruct) are updated.
- Training data: (position, question, answer) triples generated programmatically from the Hanoi solver.
- This mirrors the eventual chess setup where LC0 is the frozen encoder and the language model learns to
  interpret its hidden states without access to LC0's move-prediction head.

**Composite model class (`HanoiEncoderDecoder`):**

Wraps `self.encoder` (frozen `HanoiEncoder`) and `self.decoder` (SmolLM). Given a Hanoi state and a natural
language question, the forward pass:
1. Runs the encoder to obtain last-layer hidden states `(3, d_enc)`.
2. Passes them through `KVProjector` to produce per-layer KV tensors of sequence length 3.
3. Prepends the KV cache to the decoder's attention at every layer.
4. Runs the decoder on the tokenized question, attending to the injected position context.
5. Returns the decoder's generated answer.

**Two-stage SFT curriculum:**

Stage 1: state-tracking questions only. Three question types sampled uniformly:
- disc-location  — "What peg is disc X on?"
- peg-contents   — "What discs are on peg Y?"
- top-disc       — "What is the top disc on peg Y?" *(bridges state-tracking and move generation)*

The top-disc question type is critical for generalization: it explicitly trains the model to read the
top-of-peg disc from the encoder, which is exactly the lookup needed to correctly populate `<DISC_y>`
in a move tag for unseen k values.

Stage 2: optimal-move questions (80%) mixed with state-tracking (20%).
State-tracking mix increased from 10% to 20% to keep encoder grounding strong during move training,
reducing the risk of the model pattern-matching disc numbers from the training k distribution.

Answer format uses structured parse tags for reliable evaluation:
- disc-location  → `"...\n\n<PEG_x>"`
- peg-contents   → `"...\n\n<DISC_a><DISC_b>..."` (or `<NONE>`)
- top-disc       → `"...\n\n<DISC_y>"` (the smallest-index disc on that peg, or `<NONE>`)
- optimal-move   → `"...\n\n<PEG_x><DISC_y><PEG_z>"`

Evaluation uses tag-match accuracy (cheap) plus autonomous rollouts where the LLM controls the full
game — no teacher forcing, no oracle hints.

**Optimizer:**

Two separate parameter groups to account for the different initialization scales:
- `KVProjector` (randomly initialized): lr = 1e-3
- Decoder (pretrained SmolLM):          lr = 2e-5

Both use AdamW with weight decay 0.01 and a cosine schedule with 300 warmup steps.

**Evaluation:**

`hanoi/eval_decoder.py` runs the full autonomous rollout evaluation against a saved checkpoint:

```bash
python -m hanoi.eval_decoder --verbose
python -m hanoi.eval_decoder --k-max 4  # quick check on small disk counts only
```

Loads `hanoi/checkpoints/decoder_best.pt` by default. Prints per-k breakdown of
steps, extra moves, illegal picks, parse failures, and solved status.

### Generalization Experiment (planned)

**Motivation:** Does stage 2 teach a structural planning rule (the recursive Hanoi algorithm) that transfers
to unseen disc counts, or does it memorize patterns per k? The encoder and stage 1 together give the decoder
full board-state grounding for all k — the question is whether planning generalizes on top of that.

**Setup:** Extend to k_max=9, hold out k=5 and k=9 from stage 2 only.

| Component       | k values seen      | Purpose                                      |
|-----------------|--------------------|----------------------------------------------|
| Encoder         | k = 1..9           | Full coverage; representations for all k     |
| Decoder stage 1 | k = 1..9           | State-tracking grounding for all k           |
| Decoder stage 2 | k = 1..4, 6..8     | Planning training; k=5 and k=9 withheld      |
| Evaluation      | k = 1..9 (all)     | Test generalization on held-out k=5, k=9     |

**Expected signals:**
- k=5 (interpolation gap): the decoder has seen planning for both smaller and larger k; this is a warm-up
  check — failure here would be a red flag.
- k=9 (extrapolation): the primary test. k=9 is strictly beyond any disc count seen in stage 2, so success
  here is strong evidence the decoder learned the underlying recursive Hanoi structure rather than
  per-k move patterns. Failure means it is interpolating, not truly generalizing.

**Implementation changes needed:**
- Pass `exclude_k` list to the stage 2 data generator so those disc counts are filtered from move questions.
- Retrain encoder with `--k-max 9` and decoder with `--k-max 9`.
- Eval: run `python -m hanoi.eval_decoder --k-max 9 --verbose` and compare held-out vs. seen k results.

### Status

**Phase 1 — Encoder: DONE**
- `HanoiEncoder`: 4-layer transformer, d_model=128, trained for 3k steps
- Checkpoint: `hanoi/checkpoints/initial_k8_exp/best.pt`
- Result: all canonical starts (k=1..8) solved optimally — 8/8 solved, mean_extra=0, total_illegal=0

**Phase 2 — Decoder baseline (k=8): DONE**
- Checkpoints: `hanoi/checkpoints/initial_k8_exp/`
- Result: state_acc=1.000, move_acc=1.000; no generalization test

**Phase 3 — Generalization experiment (k=9, holdout k=5,9): DONE**
- Checkpoints: `hanoi/checkpoints/holdout59_k9_exp/`
- Result: k=5 solves (disc label occasionally wrong in narration); k=9 fails (disc label mismatch)
- See `hanoi/exp_notes.md` for full analysis

**Phase 4 — Generalization experiment with bridging (k=9, holdout k=5,9): IN PROGRESS**
- Key changes: top-disc question type in stage 1; stage 2 state-tracking mix 10%→20%; 5k stage 1 / 15k stage 2; LR reset at stage 2 boundary
- Scripts: `hanoi/scripts/train_decoder_genexp.sh`
- Checkpoints will save to: `hanoi/checkpoints/holdout59_k9_step20k_exp/`
