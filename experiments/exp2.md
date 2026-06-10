# Experiment 2 — Architecture & LR Sweep (Stage 1)

## Motivation

Exp 1 established that LLaVA clearly outperforms Flamingo and KV-proj on Skill 1. Before moving
to Skill 2, this sweep addresses two open questions:

1. **How much of LLaVA's advantage comes from the bridge vs. the decoder LoRA?**
   Exp 1 ran Flamingo with a frozen decoder and LLaVA with LoRA on all 36 decoder layers.
   These differ in two ways simultaneously (bridge architecture + decoder trainability), making
   it impossible to attribute the gap. Exp 2 runs both archs in both decoder modes so the
   effects can be separated.

2. **What is the right learning rate for cosine scheduling?**
   Exp 1 showed cosine 2e-4 is consistently better than constant 1e-4, but 2e-4 was chosen
   somewhat arbitrarily. This sweep covers 2e-4, 4e-4, 8e-4 to find the optimal starting LR.

KV-proj is dropped — it underperformed Flamingo and LLaVA in every Exp 1 configuration and
has no clear path to improvement.

---

## Flamingo Initialization

In Exp 1 the cold-start trap (α=0 → K/V gradient ≈ 0 → encoder never learned) was partially
re-engaged: alphas remained small (max |tanh(α)| ≈ 0.16–0.22) and late cross-attn layers
were dead. Exp 2 addresses this with a near-open gate init: **α initialized to 2.0**
(tanh(2) ≈ 0.964), so K/V receive ~96% of the loss gradient from step 1. W_O is still
zero-initialized so the cross-attn contribution is zero at step 0 — the frozen decoder
sees its normal input — but the gate is functionally open for learning immediately.

Unlike removing the gate entirely, keeping tanh preserves the ability for individual layers
to suppress their cross-attn output if needed (by driving α negative).

### Frozen decoder runs

All `flamingo_frozen` runs: `--alpha-init 2.0 --wo-zero-init`.

### LoRA decoder runs — two variants

When the decoder is trainable, the right gate init is less obvious, so both strategies are
tested:

| Variant | `--alpha-init` | `--wo-zero-init` | Rationale |
|---------|---------------|-----------------|-----------|
| `open`  | 2.0           | yes             | Same as frozen: gate open, clean start, full K/V gradient |
| `std`   | 0.0           | no (random W_O) | Original Flamingo default: gate closed, W_O random — alpha gets gradient via W_O immediately but K/V are initially starved |

This adds 6 runs (2 datasets × 3 LR) relative to the original 24-run design, giving **30 total**.

---

## Design

### Axes

| Axis | Values |
|------|--------|
| Architecture | Flamingo, LLaVA |
| Decoder mode | bridge-only (frozen), bridge+LoRA r16 (Flamingo: two gate init variants) |
| Dataset | v2.1, v3 |
| LR (cosine schedule) | 2e-4, 4e-4, 8e-4 |

All runs: cosine schedule, 5% warmup, effective batch 256 (bs=64, grad_accum=4), bfloat16,
**30k steps, eval every 3k steps** (10 eval points per run). Reduced from Exp 1's 50k — the
goal is architecture comparison at convergence. LLaVA-lora16 converges well within this budget;
the open question is whether LLaVA-frozen and Flamingo (with fixed gate init) also converge, and
whether higher LR accelerates this.

### Run matrix (30 jobs)

Naming convention: `stage1_{arch}_{decoder_mode}_{dataset}_{lr}`  
where `decoder_mode` is `frozen`, `lora16_open`, or `lora16_std`, and `lr` is `2e4` / `4e4` / `8e4`.

| # | Run name | Arch | Decoder | Gate init | Dataset | LR |
|---|----------|------|---------|-----------|---------|-----|
| 0 | stage1_flamingo_frozen_v2.1_2e4 | Flamingo | frozen | α=2, W_O=0 | v2.1 | 2e-4 |
| 1 | stage1_flamingo_frozen_v2.1_4e4 | Flamingo | frozen | α=2, W_O=0 | v2.1 | 4e-4 |
| 2 | stage1_flamingo_frozen_v2.1_8e4 | Flamingo | frozen | α=2, W_O=0 | v2.1 | 8e-4 |
| 3 | stage1_flamingo_frozen_v3_2e4 | Flamingo | frozen | α=2, W_O=0 | v3 | 2e-4 |
| 4 | stage1_flamingo_frozen_v3_4e4 | Flamingo | frozen | α=2, W_O=0 | v3 | 4e-4 |
| 5 | stage1_flamingo_frozen_v3_8e4 | Flamingo | frozen | α=2, W_O=0 | v3 | 8e-4 |
| 6 | stage1_flamingo_lora16_open_v2.1_2e4 | Flamingo | LoRA r16 | α=2, W_O=0 | v2.1 | 2e-4 |
| 7 | stage1_flamingo_lora16_open_v2.1_4e4 | Flamingo | LoRA r16 | α=2, W_O=0 | v2.1 | 4e-4 |
| 8 | stage1_flamingo_lora16_open_v2.1_8e4 | Flamingo | LoRA r16 | α=2, W_O=0 | v2.1 | 8e-4 |
| 9 | stage1_flamingo_lora16_open_v3_2e4 | Flamingo | LoRA r16 | α=2, W_O=0 | v3 | 2e-4 |
| 10 | stage1_flamingo_lora16_open_v3_4e4 | Flamingo | LoRA r16 | α=2, W_O=0 | v3 | 4e-4 |
| 11 | stage1_flamingo_lora16_open_v3_8e4 | Flamingo | LoRA r16 | α=2, W_O=0 | v3 | 8e-4 |
| 12 | stage1_flamingo_lora16_std_v2.1_2e4 | Flamingo | LoRA r16 | α=0, W_O=rand | v2.1 | 2e-4 |
| 13 | stage1_flamingo_lora16_std_v2.1_4e4 | Flamingo | LoRA r16 | α=0, W_O=rand | v2.1 | 4e-4 |
| 14 | stage1_flamingo_lora16_std_v2.1_8e4 | Flamingo | LoRA r16 | α=0, W_O=rand | v2.1 | 8e-4 |
| 15 | stage1_flamingo_lora16_std_v3_2e4 | Flamingo | LoRA r16 | α=0, W_O=rand | v3 | 2e-4 |
| 16 | stage1_flamingo_lora16_std_v3_4e4 | Flamingo | LoRA r16 | α=0, W_O=rand | v3 | 4e-4 |
| 17 | stage1_flamingo_lora16_std_v3_8e4 | Flamingo | LoRA r16 | α=0, W_O=rand | v3 | 8e-4 |
| 18 | stage1_llava_frozen_v2.1_2e4 | LLaVA | frozen | — | v2.1 | 2e-4 |
| 19 | stage1_llava_frozen_v2.1_4e4 | LLaVA | frozen | — | v2.1 | 4e-4 |
| 20 | stage1_llava_frozen_v2.1_8e4 | LLaVA | frozen | — | v2.1 | 8e-4 |
| 21 | stage1_llava_frozen_v3_2e4 | LLaVA | frozen | — | v3 | 2e-4 |
| 22 | stage1_llava_frozen_v3_4e4 | LLaVA | frozen | — | v3 | 4e-4 |
| 23 | stage1_llava_frozen_v3_8e4 | LLaVA | frozen | — | v3 | 8e-4 |
| 24 | stage1_llava_lora16_v2.1_2e4 | LLaVA | LoRA r16 | — | v2.1 | 2e-4 |
| 25 | stage1_llava_lora16_v2.1_4e4 | LLaVA | LoRA r16 | — | v2.1 | 4e-4 |
| 26 | stage1_llava_lora16_v2.1_8e4 | LLaVA | LoRA r16 | — | v2.1 | 8e-4 |
| 27 | stage1_llava_lora16_v3_2e4 | LLaVA | LoRA r16 | — | v3 | 2e-4 |
| 28 | stage1_llava_lora16_v3_4e4 | LLaVA | LoRA r16 | — | v3 | 4e-4 |
| 29 | stage1_llava_lora16_v3_8e4 | LLaVA | LoRA r16 | — | v3 | 8e-4 |

---

## What We Expect to Learn

**Bridge vs. decoder LoRA (frozen vs. lora16, same arch):**  
If Flamingo-lora16 ≈ LLaVA-lora16, the LoRA is doing the heavy lifting and the bridge
architecture matters less. If Flamingo-frozen << LLaVA-frozen, the MLP projection is
fundamentally better at communicating encoder information than cross-attention.

**Gate init effect (Flamingo-frozen Exp2 vs. Flamingo Exp1):**  
Exp 1 Flamingo-frozen with 2e-4 cosine reached 95.3–98.0% pc (α=0 default). If Exp 2
Flamingo-frozen at the same 2e-4 significantly exceeds this, the α=2 init is responsible.

**Open vs. std gate for LoRA decoder (flamingo_lora16_open vs. flamingo_lora16_std):**  
If `open` >> `std`, the cold-start trap is the main bottleneck for Flamingo with a hot decoder.
If they're similar, the gate init doesn't matter much when LoRA is in play (the decoder adapts
regardless).

**LR sensitivity:**  
Exp 1 showed 2e-4 is better than 1e-4. Whether the optimum is higher than 2e-4 (or whether
8e-4 causes instability) is the main open question.

---

## Optimizer Parameter Groups

Given a base learning rate `lr`, the groups seen by AdamW under each config:

| Config | bridge¹ | decoder LoRA | new\_embed |
|--------|---------|-------------|-----------|
| flamingo, frozen (`lora_rank=-1`) | `lr` | — | `lr × 0.1` |
| flamingo, lora16 (`lora_rank=16`) | `lr` | `lr` | `lr × 0.1` |
| llava, frozen (`lora_rank=-1`) | `lr` | — | `lr × 0.1` |
| llava, lora16 (`lora_rank=16`) | `lr` | `lr` | `lr × 0.1` |

¹ Flamingo bridge = `x_attn_layers`; LLaVA bridge = `connector + file_embed + rank_embed`

**In Exp 2, all `lora16` runs pass `--decoder-lr 2e-4`**, fixing the decoder LoRA group at 2e-4
regardless of the swept `--lr`. This decouples bridge and decoder learning rates so the LR sweep
cleanly reflects the bridge's optimal LR rather than a joint optimum.

Effective group LRs for `lora16` runs in this sweep:

| `--lr` (bridge) | decoder LoRA (`--decoder-lr 2e-4`) | new\_embed |
|---|---|---|
| 2e-4 | 2e-4 | 2e-5 |
| 4e-4 | 2e-4 | 4e-5 |
| 8e-4 | 2e-4 | 8e-5 |

`--decoder-lr` is a new optional argument added to `train.py`; if omitted it falls back to `--lr`
(preserving prior behavior). For full FT (`lora_rank=0`) the existing `lr × 0.1` default also
applies when `--decoder-lr` is absent.

---

## Status

**NOT STARTED**
