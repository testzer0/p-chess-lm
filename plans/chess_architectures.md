# Architecture Experiments

## Code Structure

All three model architectures live in a `chesslm/models/` subpackage:

```
chesslm/models/
    __init__.py     ← re-exports FlamingoChessLM, LLaVAChessLM, KVProjChessLM, ChessLM (Protocol)
    base.py         ← ChessLM Protocol + init_new_token_embeddings()
    flamingo.py     ← FlamingoChessLM + DenseXAttn + FFN helpers
    llava.py        ← LLaVAChessLM
    kv_proj.py      ← KVProjChessLM
```

`chesslm/model.py` and `chesslm/utils/model_utils.py` are retired; importers update their paths.

### Design Principles

- **No shared base class with implementation.** The three architectures differ too much (forward passes, decoder freezing behavior, trainable modules) to benefit from a shared `__init__` or `forward`.
- **Duck typing via Protocol.** `base.py` defines a `ChessLM` Protocol specifying the interface `train.py` depends on. Each class is fully independent.
- **Shared helpers, not shared state.** `base.py` contains pure functions called from each class's `__init__` — no inheritance, no shared state. See helper list below.
- **Decoder passed in, not constructed.** All three constructors take `(decoder, n_new_tokens, lora_rank, ...)`. Each class owns a reference to the decoder and delegates freezing/LoRA setup to the shared helpers.

### `lora_rank` semantics (all three architectures)

| `lora_rank` | Decoder treatment |
|---|---|
| `< 0` (default `-1` for Flamingo) | Fully frozen — no grad, excluded from checkpoint |
| `= 0` (default for LLaVA / KV Proj) | Fully trainable — all decoder params updated |
| `> 0` | Backbone frozen + LoRA adapters on Q/K/V/O |

### Shared helpers in `base.py`

| Helper | Purpose |
|---|---|
| `init_new_token_embeddings(n, dim)` | Returns `(new_embed, new_lm_head)` with tied weights |
| `apply_lora(decoder, lora_rank)` | Wraps decoder with PEFT LoRA if `lora_rank > 0`; otherwise no-op |
| `unwrap_decoder(decoder)` | Strips PEFT wrapper → underlying HF CausalLM |
| `decoder_trainable_params(decoder, lora_rank)` | Returns the param list to optimize ([] / all / LoRA-only) |
| `save_decoder_state(decoder, lora_rank)` | Serializes trainable decoder state (PEFT or full) |
| `load_decoder_state(decoder, lora_rank, state)` | Loads state produced by `save_decoder_state` |

### `ChessLM` Protocol (base.py)

```python
class ChessLM(Protocol):
    def forward(
        self,
        input_ids: torch.Tensor,              # (B, S)
        encoder_hidden_states: torch.Tensor,  # (B, 16, 64, 1024)
        attention_mask: torch.Tensor,         # (B, S)
    ) -> torch.Tensor: ...                    # (B, S, V)

    def trainable_parameters(self) -> Iterator[nn.Parameter]: ...
    def trainable_state_dict(self) -> dict: ...
    def load_trainable_state_dict(self, state_dict: dict) -> None: ...
    def get_diagnostics(self) -> dict[str, float]: ...
    def param_groups(self, lr: float) -> list[dict]: ...
```

`train.py` type-hints against `ChessLM` and works with any of the three implementations unchanged.

- `trainable_state_dict` / `load_trainable_state_dict`: arch-specific checkpoint save/load; `train.py` calls these without knowing which modules are trainable.
- `get_diagnostics`: arch-specific metrics logged each eval (e.g. Flamingo logs `tanh(alpha_attn/ffn)` per layer).
- `param_groups(lr)`: returns optimizer param groups with per-group learning rates. Each arch decides its own groupings. `train.py` passes this directly to `AdamW`.

---

## Experiment A: Flamingo-style Cross-Attention Bridge

**File:** `chesslm/models/flamingo.py`

**Default (`lora_rank=-1`):** LC0 encoder frozen, SmolLM3 decoder fully frozen.  
**Trainable:** 16 `DenseXAttn` sublayers + new token embeddings (~470M params).  
**With `lora_rank>0`:** decoder backbone frozen + LoRA on Q/K/V/O; x-attn layers + LoRA adapters trained.  
**With `lora_rank=0`:** decoder fully trainable alongside x-attn layers.

### Architecture

One `DenseXAttn` sublayer inserted before each of decoder layers 0, 2, 4, ..., 30 (every other layer, 16 total). X-attn sublayer `i` cross-attends to encoder layer `i` (1-to-1 pairing):

```
x-attn 0  (before dec layer  0)  ←  encoder layer  0  (embedding block)
x-attn 1  (before dec layer  2)  ←  encoder layer  1
...
x-attn 15 (before dec layer 30)  ←  encoder layer 15  (policy-shaped, high-level)
```

Injecting before layer 0 means special token embeddings attend to encoder states before any decoder self-attention runs.

### DenseXAttn sublayer

```
Attention (full MHA — fresh params):
  W_Q: (2048, 2048)   norm_y → Q
  W_K: (1024, 2048)   norm_x → K
  W_V: (1024, 2048)   norm_x → V
  W_O: (2048, 2048)   attn output → residual
  alpha_attn: scalar, tanh gate, init=0

FFN (2× hidden dim, relu² default):
  fc1: (2048, 4096)
  fc2: (4096, 2048)
  alpha_ffn: scalar, tanh gate, init=0

Per sublayer: ~29.4M params × 16 = ~470M total
```

**Tanh gate:** residual contribution is `tanh(α) * xattn_output`, α initialized to 0. X-attn starts as zero contribution and opens up gradually — prevents noise injection into the frozen backbone at the start of training.

### Forward

Manual decoder layer loop — calls `base.model.layers[i](...)` directly (via `_base_decoder`, which strips any PEFT wrapper), inserting x-attn before the appropriate layers. Does not call `self.decoder.forward()`. Builds the same 4D causal mask SmolLM3 uses internally (`create_causal_mask`).

**PEFT compatibility:** PEFT patches at the `nn.Linear` leaf level inside each layer (`q_proj`, `k_proj`, etc.). Calling `decoder_layer(h, ...)` directly still fires LoRA adapters transparently — the manual loop is fully compatible with `lora_rank > 0`.

### `trainable_parameters`

`x_attn_layers` + decoder params per `lora_rank` (see table above) + `new_embed` (lm_head tied, excluded). When `lora_rank < 0`, decoder contributes nothing to this list and is excluded from the checkpoint.

---

## Experiment B: LLaVA-style MLP Connector

**File:** `chesslm/models/llava.py`

**Frozen:** LC0 encoder, SmolLM3 decoder backbone weights.  
**Trainable:** MLP connector + positional embeddings + LoRA on decoder Q/K/V/O + new token embeddings (~83M params).

### Architecture

All 16 encoder hidden states channel-concatenated per square, projected to decoder dim via a 2-layer MLP, then prepended as 64 prefix tokens to the decoder input sequence.

```
Connector MLP:
  LayerNorm:  (16384,)         →  0.03M
  fc1 (SiLU): (16384, 4096)   → 67.11M
  fc2:         (4096, 2048)   →  8.39M

2D positional embeddings (additive, applied after MLP):
  file_embed: (8, 2048)        →  0.02M
  rank_embed: (8, 2048)        →  0.02M

LoRA (rank 16, Q/K/V/O, all 36 decoder layers):
  ~0.21M per layer × 36        →  ~7.7M

New token embeddings:           →  ~0.2M
─────────────────────────────────────────
Total trainable:                ~83.5M
```

### Forward

1. Encoder produces 16 × (B, 64, 1024) → concatenate along channel dim → (B, 64, 16384).
2. LayerNorm → fc1 → SiLU → fc2 → (B, 64, 2048).
3. Add learned 2D spatial embeddings: `prefix[i] += file_embed[file(i)] + rank_embed[rank(i)]`.
4. Prepend prefix to token embeddings; extend attention mask by 64 ones on the left.
5. RoPE assigns position **0** to every prefix token (the 64 squares share one position so RoPE imposes no relative ordering on them); text tokens occupy positions 64..63+S. Callers supply 0-based text `position_ids`; the forward offsets them by `N_ENC_SQUARES=64` internally.
6. LoRA-adapted decoder runs as a black box via `self.decoder(inputs_embeds=..., attention_mask=...)`.

**Why 2D spatial embeddings instead of 2D-RoPE:** SmolLM3 uses standard 1D RoPE baked into frozen weights. Additive learned file/rank embeddings achieve the same spatial inductive bias without modifying the attention kernel. Pinning all 64 prefix tokens to RoPE position 0 means inter-prefix attention has no positional contribution from RoPE — spatial structure flows entirely through file/rank embeddings.

**Why LoRA:** the frozen decoder was never trained to extract information from prepended prefix tokens. LoRA on Q/K/V/O allows the decoder's attention to learn to route queries toward the relevant prefix positions. Rank 16 is the starting point; ranks 8 and 32 are natural ablations.

### `trainable_parameters`

MLP connector + positional embeddings + LoRA parameters + `new_embed`.

---

## Experiment C: Direct KV Projection

**File:** `chesslm/models/kv_proj.py`

**Frozen:** LC0 encoder, SmolLM3 decoder backbone weights.  
**Trainable:** KV projector (1 LayerNorm + 36×2 linear maps) + LoRA on decoder Q/K/V/O + new token embeddings (~612M params).

### Architecture

All 16 encoder layers concatenated per square, projected to per-decoder-layer K and V tensors via independent linear maps, injected directly into `past_key_values`. No new attention sublayers, no prefix tokens in the input sequence.

```
KV Projector:
  LayerNorm (shared):     (16384,)     →   0.03M
  W_K[i] per dec layer:  (16384, 512) →   8.39M   # → n_kv_heads(4) × head_dim(128)
  W_V[i] per dec layer:  (16384, 512) →   8.39M
  36 layers total:        36 × 16.78M → 604.1M

LoRA (rank 16, Q/K/V/O, all 36 decoder layers):   →  ~7.7M
New token embeddings:                              →  ~0.2M
────────────────────────────────────────────────────────────
Total trainable:                                  ~612M
```

### Forward

1. Encoder produces 16 × (B, 64, 1024) → concatenate → (B, 64, 16384).
2. Shared LayerNorm → (B, 64, 16384).
3. For each decoder layer `i`: `K_i = W_K_i(LN_out)` → reshape → (B, 4, 64, 128); same for V.
4. Build `past_key_values` as list of `(K_i, V_i)` tuples.
5. Extend attention mask by 64 ones on the left.
6. Text tokens occupy RoPE positions 64..63+S. The injected K tensors have **no RoPE applied** — spatial structure is carried entirely by the per-layer `W_K`/`W_V` projections. Callers supply 0-based text `position_ids`; the forward offsets by `N_ENC_SQUARES=64` internally so the cache "looks like" 64 prior context positions to RoPE-trained attention heads.
7. LoRA-adapted decoder runs via `self.decoder(..., past_key_values=past_key_values)`.

**Key distinction from Exp B:** no prefix tokens enter the residual stream. Text tokens attend to 64 KV positions at every layer, but those positions have no self-attention among themselves and no residual representation. Different decoder layers independently weight different aspects of the encoder via their own W_K/W_V projections.

**No explicit positional embeddings:** spatial structure must be encoded implicitly via W_K/W_V projections. This is a potential weakness relative to Exp B's explicit file/rank embeddings.

### `trainable_parameters`

KV projector parameters + LoRA parameters + `new_embed`.

---

## Comparison

| | Exp A (Flamingo) | Exp B (LLaVA) | Exp C (KV Proj) |
|---|---|---|---|
| Trainable params (default) | ~470M (frozen dec) | ~83M (LoRA dec) | ~612M (LoRA dec) |
| Default `lora_rank` | `-1` (frozen) | `0` (full) | `0` (full) |
| Decoder options | frozen / LoRA / full | LoRA / full | LoRA / full |
| Encoder info per layer | 1-to-1 x-attn pairing | all 16 concatenated | all 16 concatenated |
| Decoder sees encoder | via new x-attn sublayers | once, as prefix tokens | via KV cache at each layer |
| New modules | 16 DenseXAttn sublayers | 1 MLP + positional embeds | 1 LayerNorm + 36×2 linears |
| Spatial bias | none explicit | learned file/rank embeds | none (implicit in W_K/W_V) |
| Prefix in residual stream | no | yes | no |
| Hierarchical info | 16 progressive injections | collapsed into 64 values | same projection at all depths |

Initial comparison runs on Skill 1 (current position understanding) will determine which architecture to prioritize for later stages.

---

## Optimizer & Scheduler

All three archs use AdamW. Param groups are arch-specific via `model.param_groups(lr)`:

All three archs share the same `param_groups(lr)` convention:

| Group | Members | LR |
|---|---|---|
| **bridge** | Flamingo: `x_attn_layers`. LLaVA: `connector + file_embed + rank_embed`. KVProj: `layer_norm + W_K + W_V`. | `lr` |
| **decoder** *(only if `lora_rank ≥ 0`)* | `lora_rank=0`: all decoder params (full backbone unfreeze). `lora_rank>0`: LoRA-A/B adapter params only. | `lora_rank>0` → `lr` (LoRA-B starts at 0; adapters need the full learning signal). `lora_rank=0` → `lr * 0.1` (the pretrained backbone is already near a good minimum; large updates destroy it). |
| **new_embed** *(only if `n_new_tokens > 0`)* | `new_embed.weight` (tied with `new_lm_head`) | `lr * 0.1` — semantic init is already near the target; large updates would destroy it. |

### Scheduler (`--scheduler`, default `constant`)

All schedulers include a linear warmup phase controlled by `--warmup-ratio` (default `0.05` = 5% of `n_steps`).

| `--scheduler` | Behavior after warmup |
|---|---|
| `constant` | lr stays at peak |
| `cosine` | cosine decay to 0 over remaining steps |
| `linear` | linear decay to 0 over remaining steps |

Warmup applies to all three scheduler types. `--warmup-ratio 0.0` disables warmup.
