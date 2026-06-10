# KV Bridge Design Document

## High level goal

Train the KV bridge so that a (potentially frozen) decoder model can successfully read and interpret the hidden states of a frozen oracle model.

## Existing Results

### Naive Midtraining

My collaborator tried to naively train the KV bridge and a 3B decoder jointly with QA pairs. This reached about 93\% accuracy after 30,000 steps of batch size 256, where the concatenated hidden states were all projected with separate Key and Value projection matrices per layer to the decoder KV cache.

I'm not exactly sure what token initialization or QA scheme that was used, but nothing that was extremely sophisticated as this was an experiment that we discussed and moved onto alternative approaches. The alternative approaches were training on best PV or best move conditioned on the projected encoder hidden states, but those had their own issues (read the KV Bridge Position Reading section in chess\_plan.md for more details), so now I am back focusing on these QA pair training.

### Sanity check with linear probes

Is the information of piece on square even linearly decodable from the hidden states? I trained a separate probe on each layer of linear state, and found that I could get around 85\% accuracy with one linear probe (1024, 13), and accuracies held approximately the same across all layers. My collaborator trained a linear probe for each square, concatenating all layers together (1024*16, 13), and was able to achieve 100\% accuracy on all squares. Thus, we conclude that this information is in fact linearly decodable but maybe not general enough for one probe.

## Next steps

### Reading from projected KV and new token initialization

I plan to initialize new tokens for all the squares and all the pieces (i.e. \<SQUARE\_E4\> or \<PIECE\_WN\>). My hypothesis is that these new tokens will essentially act as distinct linear probes for each square (or at least some subspace within the high dimensional embedding will), because when the SFT pair such as 
 - Q: What piece is on e4?
 - A: There is a black knight on e4.\<PIECE\_BN\>.
 is processed by the model, the query vector on the new token \<PIECE\_BN\> will act as a probe towards the projected hidden state (and hopefully have high attention score to the key projected by the hidden state corresponding to e4, from which the relevant value vector can be extracted and passed down from layer to layer.

The Q-K bilinear form `(W_Q @ h_dec)^T (W_K @ h_enc)` effectively gives each special token a distinct learned probe direction into the encoder's key space — analogous to the per-square probes that achieved 100% accuracy, but learned end-to-end from QA supervision.

## Decided Architecture

### Flamingo-style Cross-Attention Bridge

**Frozen components:** LC0 BT5 encoder, SmolLM3 3B decoder backbone.

**Trainable components:** 16 cross-attention sublayers only (~151M params total).

**Decoder:** SmolLM3 3B — `hidden_size=2048`, `num_attention_heads=16`, `num_key_value_heads=4`, `head_dim=128`, `num_hidden_layers=36`.

**Placement:** one x-attn sublayer inserted before each of decoder layers 0, 2, 4, ..., 30 (every other layer, covering the first 32 decoder layers; the last 4 decoder layers have no x-attn injection).

Injecting before layer 0 means the special token embeddings (`<SQUARE_E4>`, etc.) attend to encoder states before any decoder self-attention runs — the embeddings themselves learn to query the encoder.

**Encoder pairing (1-to-1):** x-attn sublayer `i` cross-attends to encoder layer `i`:
```
x-attn 0  (before dec layer  0)  ←  encoder layer  0  (embedding block)
x-attn 1  (before dec layer  2)  ←  encoder layer  1
...
x-attn 15 (before dec layer 30)  ←  encoder layer 15  (policy-shaped, high-level)
```
This progressively accumulates multi-scale encoder information in the decoder residual stream — functionally equivalent to the per-square probes that concatenated all 16 encoder layers (1024×16) to achieve 100% accuracy.

**Per x-attn sublayer parameters:**
```
Attention (full MHA — fresh params, no GQA constraint):
  W_Q: (2048, 2048)  →  4.19M    # decoder hidden → 16 Q heads × 128
  W_K: (1024, 2048)  →  2.10M    # encoder hidden → 16 KV heads × 128
  W_V: (1024, 2048)  →  2.10M
  W_O: (2048, 2048)  →  4.19M    # attn output → decoder residual
  alpha_attn: scalar (tanh), init 0

FFN (2× hidden dim, relu² activation):
  fc1: (2048, 4096)  →  8.39M
  fc2: (4096, 2048)  →  8.39M
  alpha_ffn: scalar (tanh), init 0

Per sublayer: ~29.4M × 16 = ~470M total trainable params
```

**Tanh gate:** each sublayer's output is added to the residual stream as `tanh(α) * xattn_output`, where `α` is a learned scalar initialized to 0. This ensures x-attn starts as zero contribution and opens up gradually during training, preventing noise injection into the frozen backbone at the start of training.

**Why layer 15 is included despite probe degradation:** encoder layer 15 is shaped toward LC0's policy head — poor for piece-location decoding but directly informative for move prediction, which is the downstream goal.

---

### Experiment B: LLaVA-style Connector

Motivated by the field consensus (LLaVA-1.5, InternVL2, LLaVA-OneVision) that a simple MLP projector is competitive with more complex resampler/cross-attention bridges, and by the probe result that concatenating all 16 encoder layers achieves 100% accuracy.

**Core idea:** channel-concatenate all 16 encoder hidden states per square, project to decoder dim via a 2-layer MLP, prepend the resulting 64 tokens to the decoder input sequence. No new attention sublayers — the frozen decoder's existing self-attention attends to prefix tokens alongside text tokens.

**Frozen components:** LC0 BT5 encoder, SmolLM3 3B decoder backbone weights.

**Trainable components:**
```
Connector MLP:
  LayerNorm:  (16384,)              →   0.03M
  fc1 (SiLU): (16384, 4096)        →  67.11M
  fc2:         (4096, 2048)        →   8.39M

2D positional embeddings (additive, applied after MLP):
  file_embed: (8, 2048)            →   0.02M
  rank_embed: (8, 2048)            →   0.02M

LoRA (rank 16, Q/K/V/O, all 36 decoder layers):
  per layer: 2×(2048×16 + 16×2048) + 2×(512×16 + 16×2048)  →  ~0.21M
  36 layers total                                             →   ~7.7M

New token embeddings (same as Experiment A):                  →   ~0.2M
─────────────────────────────────────────────────────────────────────────
Total trainable:                                              ~83.5M
```

**Forward pass:**
1. Encoder produces 16 × (B, 64, 1024) hidden states (one per layer).
2. Concatenate along channel dim → (B, 64, 16384).
3. LayerNorm → fc1 → SiLU → fc2 → (B, 64, 2048).
4. Add learned 2D spatial embeddings: `prefix[i] += file_embed[file(i)] + rank_embed[rank(i)]` — bakes 8×8 board structure into token values without touching the decoder's RoPE.
5. Prepend prefix to token embeddings; extend attention mask by 64 ones on the left.
6. RoPE assigns positions 0–63 to prefix tokens, text tokens continue from 64.
7. LoRA-adapted decoder runs normally — self-attention attends over prefix + text jointly.

**Why not 2D-RoPE:** SmolLM3 uses standard 1D RoPE baked into frozen weights. Applying MRoPE (as in Qwen2-VL) requires splitting head dimensions across modalities and modifying the attention kernel — incompatible with a frozen backbone. Additive learned file/rank embeddings achieve the same spatial inductive bias without any RoPE changes.

**Why LoRA:** the frozen decoder was never trained to extract information from prepended prefix tokens. LoRA on Q/K/V/O allows the decoder's attention to learn to route queries toward the relevant prefix positions while preserving the pretrained language modeling capability. Rank 16 is a standard starting point; ranks 8 and 32 are natural ablations.

**Prefix placement:** configurable — prepend before full sequence (default) or insert after system prompt. Default keeps implementation simple; after-system-prompt may be more semantically natural ("here is the board, now here is the question").

**POV canonicalization:** identical to Experiment A — hidden states un-flipped for board-absolute datasets (v2/v2.1), native ordering preserved for POV-relative dataset (v3).

**Comparison with Experiment A (Flamingo):**

| | Experiment A (Flamingo) | Experiment B (LLaVA connector) |
|---|---|---|
| Trainable params | ~470M | ~83M |
| Decoder | fully frozen | frozen + LoRA |
| Encoder info per layer | 1-to-1 x-attn pairing | all 16 layers concatenated |
| Decoder sees encoder | at each x-attn sublayer | once, as prefix tokens |
| New modules | 16 DenseXAttn sublayers | 1 MLP + positional embeds |
| Hierarchical info | accumulated across 16 injections | collapsed into 64 token values |

---

### Experiment C: Direct KV Projection

Motivated by the Hanoi encoder-decoder (`hanoi/model.py: KVProjector`), which projects encoder hidden states directly into each decoder layer's KV cache via `past_key_values` — no new attention sublayers, no prefix tokens in the input sequence. Evaluated there against a frozen Hanoi encoder (d_enc=128, 3 peg tokens) and achieved strong QA performance.

**Core idea:** concatenate all 16 LC0 encoder layers per square, apply a shared LayerNorm, then project to per-decoder-layer K and V tensors via independent linear maps. The 64 projected positions are prepended to the KV cache at every decoder layer, so the decoder's existing self-attention can route queries toward encoder-derived context at each depth — without inserting any new modules into the forward graph.

**Frozen components:** LC0 BT5 encoder, SmolLM3 3B decoder backbone weights.

**Trainable components:**
```
KV Projector:
  LayerNorm:            (16384,)              →   0.03M   # shared across all 36 layers
  W_K[i] per dec layer: (16384,  512)         →   8.39M   # 16384 → n_kv_heads(4) × head_dim(128)
  W_V[i] per dec layer: (16384,  512)         →   8.39M
  36 layers total:      36 × 16.78M           → 604.1M

LoRA (rank 16, Q/K/V/O, all 36 decoder layers):           →   ~7.7M
New token embeddings (same as Exp A/B):                   →   ~0.2M
──────────────────────────────────────────────────────────────────────
Total trainable:                                          ~612M
```

**Forward pass:**
1. Encoder produces 16 × (B, 64, 1024) hidden states (one per layer).
2. Concatenate along channel dim → (B, 64, 16384).
3. Shared LayerNorm → (B, 64, 16384).
4. For each decoder layer `i`:
   - `K_i = W_K_i(LN_out)` → reshape → `(B, 4, 64, 128)`
   - `V_i = W_V_i(LN_out)` → reshape → `(B, 4, 64, 128)`
5. Build `past_key_values` as a list of `(K_i, V_i)` tuples, passed to `decoder(..., past_key_values=...)`.
6. Extend attention mask by 64 ones on the left (text tokens attend to all 64 KV positions).
7. LoRA-adapted decoder runs normally — Q vectors at each layer attend over 64 projected positions + text context.

**Distinction from Experiment B (prefix tokens):**

In Experiment B, 64 prefix tokens are prepended to the input sequence. Those tokens enter the residual stream and participate in causal self-attention: prefix tokens attend to each other, text tokens attend to all prefix positions. The same prefix representation flows through all 36 layers (though transformed by each layer's FFN/SA).

In Experiment C, there are no prefix tokens in the input sequence. Each decoder layer independently receives its own `(K_i, V_i)` computed from the same LayerNorm output — different layers can selectively weight different aspects by learning distinct W_K/W_V projections. Text tokens attend to the 64 KV positions, but those positions have no self-attention among themselves and no residual stream representation. This is strictly a key-value injection, not a token injection.

**Distinction from Experiment A (Flamingo):**

Experiment A uses fresh cross-attention sublayers (DenseXAttn) with their own Q/K/V/O projections and alpha gates, inserted before decoder layers. Each x-attn sublayer cross-attends to a single paired encoder layer. The output is added to the decoder residual stream via `tanh(α) * xattn_out`.

Experiment C repurposes the decoder's existing self-attention by injecting encoder-derived KV positions via `past_key_values`. No new attention operations — the decoder's own Q vectors (LoRA-modified) simply see 64 additional KV positions as if they were cached past tokens. All 16 encoder layers are visible at every decoder depth via the concatenated projection.

**Why no positional embeddings:** the projected KV positions are prepended before token position 0. The decoder's RoPE assigns relative positions to these slots automatically — the text tokens "look back" at them with a learned offset. Unlike Experiment B where prefix tokens in the input sequence benefit from explicit 2D spatial embeddings, here the spatial structure must be encoded implicitly via the W_K/W_V projections themselves (which map different encoder-layer orderings and per-square features to KV space). This may be a weakness compared to Experiment B's explicit file/rank embeddings.

**Comparison across all experiments:**

| | Experiment A (Flamingo) | Experiment B (LLaVA) | Experiment C (KV Proj) |
|---|---|---|---|
| Trainable params | ~470M | ~83M | ~612M |
| Decoder | fully frozen | frozen + LoRA | frozen + LoRA |
| Encoder info per layer | 1-to-1 x-attn pairing | all 16 concatenated | all 16 concatenated |
| Decoder sees encoder | via new x-attn sublayers | once, as prefix tokens | via KV cache at each layer |
| New modules | 16 DenseXAttn sublayers | 1 MLP + positional embeds | 1 LayerNorm + 36×2 linears |
| Spatial bias | none explicit | learned file/rank embeds | none (implicit in W_K/W_V) |
| Hierarchical info | accumulated across 16 injections | collapsed into 64 token values | same projection seen at all depths |
| Prefix in residual stream | no (cross-attn, not prefix) | yes | no (KV-only, no residual) |
