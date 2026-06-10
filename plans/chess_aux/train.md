# Stage 1 Midtraining: train.py Design

## Overview

Train the FlamingoChessLM bridge (x_attn_layers only) on board QA pairs.
Encoder and decoder are frozen; only DenseXAttn layers + new token embeddings are trained.

---

## Code layout

Shared constants live in `chesslm/utils/utils.py` (token defs, `_generate_position_dict`,
`SYSTEM_PROMPT`, `encode_positions`) and are imported from there by all other modules.
This avoids circular imports between `training_utils` ↔ `eval_utils`.

```
chesslm/utils/utils.py          ← ANSWER_SPECIAL_TOKENS, SYSTEM_PROMPT, encode_positions
chesslm/utils/training_utils.py ← collate_fn, init_*, initialize_training_objects
chesslm/utils/eval_utils.py     ← _batched_decode, run_eval
chesslm/train.py                ← parse_args, main loop
```

---

## Tokenization

Done **on-the-fly in `collate_fn`** (not pre-tokenized). Sequences are short (~50–80 tokens),
parallelizes across DataLoader workers at negligible cost, and the format stays easy to iterate.

### Chat format

SmolLM3 instruct format with `/no_think /system_override` to bypass the date/metadata boilerplate
and disable the thinking block. The `<think>\n\n</think>` stub still appears before the assistant
content (unavoidable from the template) but is masked out in labels.

```
<|im_start|>system
You are ChessLM... /no_think /system_override<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<think>

</think>
{answer}<|im_end|>
```

### Label masking

Apply loss only on the assistant answer tokens (answer text + parse tag + `<|im_end|>`).
Everything before the answer — system prompt, user turn, `<|im_start|>assistant\n<think>\n\n</think>\n` —
is masked with `-100`.

Implementation: two `apply_chat_template` calls — full sequence and prompt-only with
`add_generation_prompt=True` — the length difference gives the exact label boundary.

### New token embeddings (split embedding architecture)

`n_new_tokens` new tokens (`ANSWER_SPECIAL_TOKENS`) are added to the tokenizer.
`n_new_tokens` is computed dynamically as `len(tokenizer) - orig_vocab_size` after
`tokenizer.add_tokens(...)`, so adding more special tokens in the future requires no code changes.

**Pretrained vocab is never touched.** New tokens are handled by separate trainable modules:
- `model.new_embed   : nn.Embedding(n_new_tokens, DECODER_DIM)`
- `model.new_lm_head : nn.Linear(DECODER_DIM, n_new_tokens, bias=False)`
- `new_lm_head.weight = new_embed.weight` — tied (matches SmolLM3's own `tie_word_embeddings=True`)

Forward routing: `token_id < frozen_vocab` → frozen `embed_tokens`; `token_id >= frozen_vocab` →
`new_embed` with offset. Output logits: `cat([frozen_lm_head(h), new_lm_head(h)], dim=-1)`.

### Embedding initialization strategies

Controlled by `--embed-init {semantic, random}`.

**semantic** (default): initialize new token rows from existing pretrained embeddings
- `<SQUARE_XY>` → mean of embeddings for file char ("e") and rank char ("4")
- `<PIECE_ZZ>`  → mean of embeddings for color word ("white") and piece word ("knight")
- `<EMPTY>`     → embedding of "empty"
- Reads only from frozen `embed_tokens`; writes only to `new_embed.weight` (tied write covers lm_head)

**random**: no-op — keep the default random init.

---

## Collation

Dynamic: pad to the longest sequence **in each batch** (not a global fixed length).
Right-padding — real tokens land at positions 0..L-1 in `torch.arange(S)`, so RoPE positions
are correct. `max_seq_len` arg is a safety truncation cap only, not a padding target.

### Returned keys

```
input_ids       (B, max_len)   long
attention_mask  (B, max_len)   long
labels          (B, max_len)   long   — -100 on prompt tokens and pad positions
start_fens      list[str]
moves           list[list[str]]
fens            list[str]
question_types  list[str]
answer_classes  list[list[str]]
```

---

## Encoder Encoding

`encode_positions(encoder, start_fens, moves_list, end_fens, device, dtype) → (B, 16, 64, 1024)`

- `@torch.no_grad()` — encoder is always frozen
- Build input planes: `encoder.input_planes_from_fen(start_fen, moves)` per example, stack → `(B, 112, 64)`
- Run encoder with `output_hidden_states=True`, stack `all_hidden_states` along dim=1 → `(B, 16, 64, 1024)`
- POV-canonicalize: for positions where `chess.Board(end_fen).turn == BLACK`, reindex squares with
  `sq ^ 56` so index 0 = a1 from white's perspective

---

## Loss

Standard NTP cross-entropy on answer tokens only.
`labels` already has `-100` on prompt/pad tokens so `F.cross_entropy(..., ignore_index=-100)` handles masking.
Shift: `logits[:, :-1]` vs `labels[:, 1:]` (predict next token). `F.cross_entropy` does not shift
automatically — this is our responsibility.

---

## Eval

**Generative** (not teacher-forced) — more meaningful, slower. Run every `eval_freq` optimizer steps.
Lives in `chesslm/utils/eval_utils.py`.

### Generation (`_batched_decode`)

Manual autoregressive loop — no KV cache (see Future Work). Each step runs the full sequence
through the model and indexes `attn_mask.sum(dim=1) - 1` to get the last real token's logits.

Sampling is controlled by three args:
- `--temperature 0` (default) → greedy argmax, bypasses all sampling
- `--temperature > 0` → temperature scale → top-k mask → top-p nucleus filter → multinomial sample

Order matches transformers' `LogitsWarper` pipeline exactly.

### Metrics (per question type)

- `parse_valid`     — generated output contains a well-formed parse tag
- `correct`         — parse tag matches ground-truth `answer_class` exactly
- `fen_consistent`  — parse tag is self-consistent with the board FEN

### Kernels

- `DenseXAttn` uses `F.scaled_dot_product_attention` → dispatches to FlashAttention 2 / efficient SDPA
- Decoder layers use the same path internally (HuggingFace SmolLM3)
- No kernel performance is being left on the table; O(n²) generation is the only inefficiency

---

## Training Loop

### Initialization (`initialize_training_objects`)

1. `init_model_and_tokenizer(args)` — loads `FlamingoChessLM` + LC0 encoder, extends tokenizer,
   computes `n_new_tokens` dynamically, calls `init_special_token_embeddings`
2. `init_datasets_and_dataloader(args, tokenizer)` — `load_from_disk`, wraps train set in DataLoader
   with `collate_fn`; eval dataset returned raw for generative eval
3. `init_optimizer_and_scheduler(args, model)` — two AdamW param groups:
   `x_attn_layers` at `lr`, `new_embed` at `lr * 0.1`; cosine schedule with
   `warmup_steps = int(warmup_ratio * n_steps)`

### Main loop

```python
train_iter = cycle(train_loader)
optimizer.zero_grad()

for step in range(start_step, args.n_steps):
    for _ in range(args.grad_accum_steps):
        batch      = next(train_iter)
        enc_hidden = encode_positions(...)
        with autocast(...):
            logits = model(input_ids, enc_hidden, attention_mask)
            loss   = ntp_loss(logits, labels) / args.grad_accum_steps
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
    scaler.step(optimizer); scaler.update(); scheduler.step()
    optimizer.zero_grad()

    if (step + 1) % args.eval_freq == 0:
        # do_eval: run_eval → save generations → log metrics → restore train mode
        # save_checkpoint: x_attn_layers + new_embed + optimizer + scheduler + step
```

### Output structure

```
runs/{exp_name}/
  metrics.jsonl              ← one JSON object per line (machine-parsable)
  metrics.txt                ← human-readable aligned key=value table
  step_0000500/
    checkpoint.pt            ← x_attn_layers, new_embed, optimizer, scheduler, step
    generations.json         ← [{qt, question, generated, gt_class, pred_toks, correct}, ...]
  step_0001000/
    ...
```

### Notes

- `n_steps` = optimizer steps (not forward passes)
- `model.decoder.eval()` always — frozen decoder has no dropout but we keep it explicit
- `--eval-at-start` flag (default off) — initial eval skipped because untrained generation is very slow
- `--compile` flag — wraps model with `torch.compile(mode="reduce-overhead")` after init;
  ~2 min cold-start on first forward, meaningful steady-state speedup for long runs

---

## Future Work

### KV cache for generation

Currently `model.forward()` passes `use_cache=False` and recomputes the full sequence every
generation step → O(n²) in output length. For longer sequences this becomes the bottleneck.

Planned approach:
1. **Pre-project encoder K/V once** before the generation loop — `W_K(enc_hidden)` and
   `W_V(enc_hidden)` are fixed for a given position, so only the query changes per step.
   Add a `precompute_encoder_kv()` method to `FlamingoChessLM`.
2. **Thread `past_key_values`** through `model.forward()` for decoder self-attention — set
   `use_cache=True` in each `decoder_layer(...)` call and return the updated cache.
3. Generation loop then only processes the single new token per step instead of the full sequence.

This is a meaningful refactor of `model.forward()` but architecturally clean — x-attn and
decoder KV cache are independent problems solved separately.
