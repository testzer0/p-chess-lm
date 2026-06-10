# ChessLM Experiments

Stage 1 goal: train a bridge between the LC0 encoder hidden states and a frozen SmolLM3 decoder. Evaluated via static QA — piece location and square occupancy queries (Skill 1).

---

## stage1_v1

**Run:** `chesslm/runs/stage1_v1/`
**Dataset:** `chesslm/datasets/v1/` — 1M train / 7,600 eval (100 positions × 76 questions)
**Steps:** 10,000 | **Batch:** 32 | **Grad accum:** 4 | **Effective batch:** 128
**LR:** 1e-4 cosine decay (5% warmup) | **dtype:** bfloat16

### Results

| step | loss | sq/parse | sq/fen_consistent | pc/parse | pc/fen_consistent |
|------|------|----------|-------------------|----------|-------------------|
| 500  | 0.346 | 7.3%   | 3.3%              | 19.7%    | 7.7%              |
| 1000 | 0.216 | 100%   | 42.2%             | 90.0%    | 42.7%             |
| 5000 | 0.149 | 100%   | ~65%              | 94.7%    | ~50%              |
| 10000| 0.150 | 100%   | 65.7%             | 94.7%    | 52.0%             |

`correct` showed 0% throughout — but this was entirely a **dataset/eval bug** (see below). The `fen_consistent` metric (model generates a piece token that actually exists on the board) shows genuine learning.

### Alpha gate values (tanh scale)

| step | max \|alpha_attn\| | max \|alpha_ffn\| |
|------|---------------------|------------------|
| 500  | 0.011               | 0.008            |
| 2500 | 0.024               | 0.017            |
| 10000| 0.024               | 0.017            |

Alphas plateau at step ~2500 and never move again. The cross-attention contributes ≤2.4% to the residual stream throughout training.

### K/V weight norm change (step 500 → 10000)

| weight | % change |
|--------|----------|
| W_K    | 0–4%     |
| W_V    | 0–3.4%   |
| W_Q    | 0.5–12%  |
| W_O    | 0.5–11%  |

K and V barely moved. Q and O learned significantly more.

### Bugs discovered

**1. `answer_class` format wrong (critical — invalidated all correctness metrics)**

The eval comparison expected two tokens (`['<SQUARE_XY>', '<PIECE_XY>']` for static_square, `['<PIECE_XY>', '<SQUARE_XY>']` for static_piece) but the dataset stored only one. Every prediction was marked incorrect regardless of model output. The generated text and `fen_consistent` metric are still valid — only `correct` is meaningless for v1.

Fixed in `chesslm/utils/generate_sft_data.py`: all 4 sampling locations now include the leading square/piece token in `answer_class`.

**2. Alpha cold-start trap (critical — encoder learned nothing)**

Flamingo initializes gates at 0 to protect a fine-tuned LM from random cross-attention noise. But our decoder is frozen, so the protection is unnecessary. The LM quickly found a solution that ignores the encoder entirely (high parse rate, low loss, no encoder needed). Once there, `dL/d(alpha)` ≈ 0 and K/V gradients are scaled by tanh(alpha) ≈ 0.02 — K/V learn 50× slower and receive essentially no signal.

Fixed: `alpha_attn` and `alpha_ffn` initialized to `atanh(0.5) ≈ 0.549` so gates start at 0.5. W_O zero-initialized so the cross-attention output is zero at step 0 even with non-zero alpha, preventing decoder corruption while allowing gradients to flow through K/V from the start.

**3. `answer_class` for empty/absent had wrong token prefix**

`static_square` empty answers stored `['<EMPTY>']` instead of `['<SQUARE_XY>', '<EMPTY>']`. `static_piece` absent answers stored `['<EMPTY>']` instead of `['<PIECE_XY>', '<EMPTY>']`. Related to bug 1 — same fix.

**4. Dataset path wrong**

`DATASETS_PATH` in `create_sft_dataset.py` went up too many parent levels (`parent.parent.parent / "datasets"`) landing in `chesslm/datasets/` instead of `chesslm/chesslm/datasets/`. Datasets were generated to the wrong location. Fixed to `parent.parent / "datasets"`.

**5. `itertools.cycle` killed shuffle**

Training loader was wrapped with `itertools.cycle`, which caches epoch 1 batches and replays them forever. Replaced with `_infinite_loader` (`while True: yield from loader`) which re-runs the sampler each epoch.

**6. `GradScaler` enabled for bfloat16**

`enabled=(amp_dtype != torch.float32)` triggered for bf16, which has fp32 dynamic range and doesn't need loss scaling (triggers a warning and is a no-op but wasteful). Fixed to `enabled=(amp_dtype == torch.float16)`.

**7. Eval correctness was 0% due to left-padding bug**

`_batched_decode` used right-padding, causing generated tokens to get wrong RoPE positions (real_len + step instead of max_prompt + step). Fixed to left-pad all prompts so real tokens always end at position S-1; `position_ids` computed from `attn_mask.cumsum(-1) - 1`; last logit always at `[:, -1, :]`.

**8. Cosine LR decay (minor)**

With 10k steps and an already-small warmup, cosine decay was cutting LR significantly by the end. Changed to flat 1e-4 for v2 to keep the encoder learning at full rate throughout.

---

## stage1_v2 (complete at step 40k, still running)

**Dataset:** `chesslm/datasets/v2/` — 10M train / 7,600 eval, fixed `answer_class` format
**Steps:** 50,000 | **Effective batch:** 256 (bs=64, grad_accum=4) | **LR:** 1e-4 flat
**Changes vs v1:** alpha init `atanh(0.5)`, W_O zero-init, flat LR, 10× more data

### Results

| step  | loss   | sq/correct | sq/fen_consistent | pc/correct | pc/fen_consistent |
|-------|--------|------------|-------------------|------------|-------------------|
| 5000  | 0.1176 | 83.3%      | 83.3%             | 84.3%      | 84.3%             |
| 10000 | 0.1039 | 93.7%      | 93.7%             | 90.7%      | 91.3%             |
| 15000 | 0.1045 | 96.0%      | 96.0%             | 94.0%      | 94.7%             |
| 20000 | 0.0983 | 97.2%      | 97.2%             | 96.3%      | 96.3%             |
| 25000 | 0.0974 | 97.7%      | 97.7%             | 96.0%      | 96.0%             |
| 30000 | 0.1001 | 97.9%      | 97.9%             | 97.3%      | 97.3%             |
| 35000 | 0.0990 | 98.7%      | 98.7%             | 97.7%      | 97.7%             |

parse_valid = 100% throughout. At step 10k `pc/fen_consistent` slightly exceeds `pc/correct` — model occasionally names a piece that exists on the board but at the wrong square.

### Alpha gate values

All alphas log as exactly 0.5000 throughout — **logging bug**: parameters are bfloat16 and `tanh(alpha).item()` in bf16 rounds to 0.5. Checkpoint inspection at step 5k confirmed alpha has moved to `0.5508` (from init `0.5493`), with non-zero AdamW `exp_avg`. Fixed in `train.py` (cast to float32 before tanh in `get_alpha_metrics`); takes effect in future runs.

### Alpha bootstrapping issue

Alpha receives zero gradient at init because its gradient path is `dL/dy · (1−tanh²(α)) · W_O(ctx)`, and W_O is zero-initialized so `W_O(ctx)=0`. W_O does get gradients (path is `tanh(α) · dL/dy · ctx`, non-zero since α=0.5), so W_O bootstraps first and alpha follows as W_O grows. The model learns well regardless (98.7% sq at step 35k), but alphas may plateau near 0.5 since the task becomes solvable with fixed gates.

---

## stage1_v2_flamingo_init (planned)

**Dataset:** `chesslm/datasets/v2/` — same as v2 for clean comparison
**Changes vs v2:** alpha=0, W_O random (now the default; v2 used `--alpha-init 0.5493 --wo-zero-init`)
**Also:** alpha params kept in fp32 to fix bf16 underflow — updates of ~1e-6 were smaller than the bf16 quantum (~0.004), so alphas never moved in v2/v2.1/v3.
**Hypothesis:** with random W_O, alpha gets a non-zero gradient from step 0 (`dL/dy · W_O(ctx) ≠ 0`), breaking the bootstrapping lag. In the Flamingo paper, gates grow to near 1. With our frozen decoder, the cold-start trap could re-emerge (decoder might solve the task without encoder before alpha opens), but board-state QA requires the encoder so the trap may not apply here.
**Launch command:**
```bash
EXP_NAME=stage1_v2_flamingo_init DATASET_VERSION=v2 \
sbatch chesslm/scripts/train_stage1.sh
```

---

## stage1_v2.1 (complete at step 40k)

**Dataset:** `chesslm/datasets/v2.1/` — same positions as v2; special tokens in question AND answer prose
**Hypothesis:** embedding the query token (`<SQUARE_A1>`) directly in the question tightens the association between the query and the parse tag, reducing the mapping the model must learn.

### Results

| step  | sq/correct | pc/correct |
|-------|------------|------------|
| 5000  | 94.1%      | 88.3%      |
| 10000 | 97.6%      | 95.3%      |
| 15000 | 97.7%      | 97.3%      |
| 20000 | 98.3%      | 97.0%      |
| 25000 | 98.8%      | 98.3%      |
| 30000 | 99.0%      | 98.7%      |
| 35000 | 99.1%      | 98.7%      |
| 40000 | 99.0%      | 99.3%      |

parse_valid = 100% throughout (after eval bug fix — see below).

### Eval bug (parse_valid = 0% in logged metrics)

`new_tok_in_query=True` means answer prose contains special tokens (e.g. "on `<SQUARE_E4>`.") in addition to the parse tag. The eval was collecting ALL special tokens from generated IDs, picking up the prose token as an extra prefix and making `len(pred_toks) == 3` instead of 2 — failing the `len(toks) == 2` check in `_is_valid_parse_tag`. Logged metrics showed 0% parse_valid and 0% correct throughout, even though the model was generating perfectly correct output.

Fixed in `eval_utils.py`: decode generated text, split on `\n\n`, re-encode just the parse-tag region, then filter for special tokens. The logged metrics for this run are wrong; correct numbers are in the table above (recomputed from `generations.json`).

---

## stage1_v3 (complete at step 40k)

**Dataset:** `chesslm/datasets/v3/` — POV-relative `<SQUARE_1>`…`<SQUARE_64>` tokens; no hidden-state flip
**Hypothesis:** LC0 computes hidden states from the current player's POV. Un-flipping for black may scramble spatial structure the encoder built up internally. Keeping native ordering means hidden state index `i` is always queried by the token that corresponds to what LC0 "thought about" at position `i`.
**Token init:** `<SQUARE_N>` initialized to the same embedding as `<SQUARE_XY>` for board square `N-1` (white-POV correspondence), preserving spatial structure at init.

### Results

| step  | sq_pov/correct | pc_pov/correct |
|-------|----------------|----------------|
| 5000  | 93.2%          | 91.7%          |
| 10000 | 97.0%          | 97.3%          |
| 15000 | 97.4%          | 98.7%          |
| 20000 | 98.4%          | 98.3%          |
| 25000 | 98.9%          | 98.7%          |
| 30000 | 99.0%          | 97.7%          |
| 35000 | 98.8%          | 98.3%          |
| 40000 | 99.1%          | 99.0%          |

parse_valid = 100% throughout (after eval bug fix — same issue as v2.1). Logged metrics showed 0% parse_valid; correct numbers recomputed from `generations.json`.

---

## Stage 1 Architecture Comparison

After establishing baselines on Flamingo (stage1_v2.x, stage1_v3 above), a full sweep was run comparing all three bridge architectures across both datasets and two LR schedules. All runs used bs=64, grad_accum=4 (effective batch 256), bfloat16, 50k steps.

**Baseline (`train_stage1_array.sh`):** lr=1e-4, constant schedule, 36h wall time.  
**High-LR (`train_stage1_array_hilr.sh`):** lr=2e-4, cosine decay, 24h wall time. Runs marked † hit the wall time before 50k steps.

### Summary table (sorted by pc/correct at final step)

| Run | Dataset | LR / Sched | Steps | sq/correct | pc/correct | loss |
|-----|---------|-----------|-------|-----------|-----------|------|
| stage1_llava_v2.1 | v2.1 | 1e-4 const | 50k | 100.0% | 100.0% | 0.1436 |
| stage1_llava_v2.1_hilr | v2.1 | 2e-4 cosine | 35k† | 100.0% | 100.0% | 0.0922 |
| stage1_llava_v3_hilr | v3 | 2e-4 cosine | 35k† | 100.0% | 100.0% | 0.0980 |
| stage1_flamingo_v3_hilr | v3 | 2e-4 cosine | 50k | 98.5% | 98.0% | 0.1167 |
| stage1_kv_proj_channel_concat_v3_hilr | v3 | 2e-4 cosine | 45k† | 98.4% | 97.7% | 0.1087 |
| stage1_flamingo_v2.1_hilr | v2.1 | 2e-4 cosine | 50k | 97.9% | 97.0% | 0.1161 |
| stage1_kv_proj_interleaved_v3_hilr | v3 | 2e-4 cosine | 50k | 97.9% | 97.0% | 0.1002 |
| stage1_llava_v3 | v3 | 1e-4 const | 50k | 99.4% | 97.0% | 0.1758 |
| stage1_flamingo_v2.1 | v2.1 | 1e-4 const | 50k | 98.5% | 95.3% | 0.1642 |
| stage1_flamingo_v3 | v3 | 1e-4 const | 50k | 96.9% | 95.3% | 0.1691 |
| stage1_kv_proj_channel_concat_v3 | v3 | 1e-4 const | 50k | 97.9% | 95.0% | 0.1628 |
| stage1_kv_proj_interleaved_v2.1_hilr | v2.1 | 2e-4 cosine | 50k | 95.6% | 93.0% | 0.1019 |
| stage1_kv_proj_channel_concat_v2.1_hilr | v2.1 | 2e-4 cosine | 45k† | 95.9% | 91.0% | 0.1127 |
| stage1_kv_proj_interleaved_v3 | v3 | 1e-4 const | 50k | 95.0% | 91.3% | 0.1583 |
| stage1_kv_proj_interleaved_v2.1 | v2.1 | 1e-4 const | 50k | 93.0% | 88.0% | 0.1553 |
| stage1_kv_proj_channel_concat_v2.1 | v2.1 | 1e-4 const | 50k | 91.2% | 88.3% | 0.1628 |

### Conclusions

**LLaVA is the clear winner.** It reaches 100% on both question types with v2.1 (baseline by step 20k, hilr by step 25k). Even on v3 with the harder dataset it saturates at 100% by step 25–35k. No other architecture reaches 100%.

**Higher LR (2e-4 cosine) consistently helps.** Every architecture converges faster and usually to a higher final number with hilr. The improvement is largest for Flamingo (+2–3 pp) and smaller for LLaVA (already near ceiling).

**v3 ≥ v2.1 for non-LLaVA architectures, especially early.** Flamingo and KV-proj converge faster on v3 at early steps. By 50k the gap is small. For LLaVA, v2.1 is slightly better.

**KV-proj channel_concat > interleaved** by ~2–4 pp across all LR/dataset combinations. Both lag behind Flamingo and LLaVA by ~5–10 pp.

**Flamingo alpha gates remain small.** With fp32 logging fixed, alphas in the architecture comparison runs reach max |tanh(α)| ≈ 0.16–0.22 (vs ≤0.024 in stage1_v1). The cross-attention contributes ≤22% to the residual — the frozen decoder backbone carries most of the signal.

---

## stage1_flamingo_v2.1

**Dataset:** v2.1 | **LR:** 1e-4 const | **Steps:** 50k

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 56.3% | 45.3% | 0.2373 |
| 10000 | 72.5% | 52.3% | 0.2038 |
| 15000 | 94.0% | 87.7% | 0.1742 |
| 20000 | 96.7% | 89.0% | 0.1680 |
| 25000 | 97.0% | 92.7% | 0.1656 |
| 30000 | 97.8% | 95.3% | 0.1680 |
| 35000 | 97.7% | 94.7% | 0.1681 |
| 40000 | 98.1% | 94.7% | 0.1652 |
| 45000 | 98.3% | 97.3% | 0.1671 |
| 50000 | 98.5% | 95.3% | 0.1642 |

---

## stage1_flamingo_v2.1_hilr

**Dataset:** v2.1 | **LR:** 2e-4 cosine | **Steps:** 50k

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 65.8% | 48.0% | 0.1805 |
| 10000 | 92.3% | 83.7% | 0.1365 |
| 15000 | 95.7% | 91.3% | 0.1251 |
| 20000 | 97.3% | 92.7% | 0.1194 |
| 25000 | 97.9% | 94.3% | 0.1168 |
| 30000 | 98.2% | 97.0% | 0.1166 |
| 35000 | 97.7% | 97.3% | 0.1214 |
| 40000 | 97.9% | 97.0% | 0.1167 |
| 45000 | 97.8% | 97.0% | 0.1181 |
| 50000 | 97.9% | 97.0% | 0.1161 |

---

## stage1_flamingo_v3

**Dataset:** v3 | **LR:** 1e-4 const | **Steps:** 50k

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 85.0% | 76.3% | 0.1898 |
| 10000 | 94.3% | 87.7% | 0.1843 |
| 15000 | 95.3% | 92.3% | 0.1764 |
| 20000 | 97.1% | 93.7% | 0.1746 |
| 25000 | 97.2% | 95.0% | 0.1784 |
| 30000 | 97.4% | 95.3% | 0.1704 |
| 35000 | 97.5% | 96.0% | 0.1629 |
| 40000 | 98.6% | 95.3% | 0.1715 |
| 45000 | 97.8% | 96.7% | 0.1704 |
| 50000 | 96.9% | 95.3% | 0.1691 |

---

## stage1_flamingo_v3_hilr

**Dataset:** v3 | **LR:** 2e-4 cosine | **Steps:** 50k

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 90.1% | 82.3% | 0.1348 |
| 10000 | 94.9% | 88.3% | 0.1276 |
| 15000 | 96.9% | 92.7% | 0.1232 |
| 20000 | 97.6% | 96.7% | 0.1204 |
| 25000 | 98.3% | 96.0% | 0.1177 |
| 30000 | 98.4% | 97.0% | 0.1144 |
| 35000 | 98.2% | 98.0% | 0.1150 |
| 40000 | 98.4% | 98.0% | 0.1182 |
| 45000 | 98.5% | 98.0% | 0.1179 |
| 50000 | 98.5% | 98.0% | 0.1167 |

---

## stage1_kv_proj_channel_concat_v2.1

**Dataset:** v2.1 | **LR:** 1e-4 const | **Steps:** 50k

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 67.2% | 40.3% | 0.2571 |
| 10000 | 65.6% | 49.0% | 0.2210 |
| 15000 | 78.3% | 67.3% | 0.1907 |
| 20000 | 77.8% | 76.7% | 0.1768 |
| 25000 | 87.0% | 82.0% | 0.1794 |
| 30000 | 85.3% | 85.0% | 0.1756 |
| 35000 | 90.5% | 83.7% | 0.1666 |
| 40000 | 88.9% | 84.7% | 0.1597 |
| 45000 | 90.6% | 89.7% | 0.1676 |
| 50000 | 91.2% | 88.3% | 0.1628 |

---

## stage1_kv_proj_channel_concat_v2.1_hilr

**Dataset:** v2.1 | **LR:** 2e-4 cosine | **Steps:** 45k† (24h wall time)

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 67.0% | 44.7% | 0.1830 |
| 10000 | 65.7% | 51.3% | 0.1659 |
| 15000 | 69.1% | 55.0% | 0.1428 |
| 20000 | 75.8% | 67.7% | 0.1356 |
| 25000 | 91.4% | 79.7% | 0.1263 |
| 30000 | 92.2% | 83.0% | 0.1181 |
| 35000 | 92.3% | 83.3% | 0.1186 |
| 40000 | 94.2% | 88.7% | 0.1106 |
| 45000 | 95.9% | 91.0% | 0.1127 |

---

## stage1_kv_proj_channel_concat_v3

**Dataset:** v3 | **LR:** 1e-4 const | **Steps:** 50k

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 88.6% | 63.3% | 0.2347 |
| 10000 | 89.3% | 74.0% | 0.1935 |
| 15000 | 95.6% | 81.7% | 0.1812 |
| 20000 | 94.5% | 87.0% | 0.1784 |
| 25000 | 95.8% | 86.3% | 0.1723 |
| 30000 | 94.2% | 84.7% | 0.1747 |
| 35000 | 96.7% | 88.7% | 0.1631 |
| 40000 | 96.6% | 93.3% | 0.1618 |
| 45000 | 97.9% | 93.0% | 0.1699 |
| 50000 | 97.9% | 95.0% | 0.1628 |

---

## stage1_kv_proj_channel_concat_v3_hilr

**Dataset:** v3 | **LR:** 2e-4 cosine | **Steps:** 45k† (24h wall time)

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 86.4% | 70.0% | 0.1701 |
| 10000 | 92.4% | 84.7% | 0.1354 |
| 15000 | 89.9% | 81.7% | 0.1363 |
| 20000 | 96.7% | 95.3% | 0.1184 |
| 25000 | 97.4% | 95.7% | 0.1215 |
| 30000 | 98.3% | 96.7% | 0.1176 |
| 35000 | 98.0% | 97.7% | 0.1100 |
| 40000 | 98.3% | 97.7% | 0.1081 |
| 45000 | 98.4% | 97.7% | 0.1087 |

---

## stage1_kv_proj_interleaved_v2.1

**Dataset:** v2.1 | **LR:** 1e-4 const | **Steps:** 50k

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 30.5% | 51.3% | 0.3033 |
| 10000 | 44.1% | 69.7% | 0.2520 |
| 15000 | 66.3% | 74.3% | 0.1995 |
| 20000 | 81.3% | 78.0% | 0.1913 |
| 25000 | 86.6% | 79.7% | 0.1797 |
| 30000 | 87.9% | 83.3% | 0.1706 |
| 35000 | 90.2% | 83.7% | 0.1691 |
| 40000 | 91.4% | 84.7% | 0.1613 |
| 45000 | 93.4% | 86.0% | 0.1586 |
| 50000 | 93.0% | 88.0% | 0.1553 |

---

## stage1_kv_proj_interleaved_v2.1_hilr

**Dataset:** v2.1 | **LR:** 2e-4 cosine | **Steps:** 50k

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 52.3% | 59.3% | 0.1931 |
| 10000 | 85.1% | 77.3% | 0.1395 |
| 15000 | 91.3% | 82.7% | 0.1210 |
| 20000 | 91.0% | 85.3% | 0.1184 |
| 25000 | 92.9% | 90.0% | 0.1121 |
| 30000 | 94.8% | 90.3% | 0.1110 |
| 35000 | 94.9% | 90.3% | 0.1030 |
| 40000 | 95.8% | 92.3% | 0.1018 |
| 45000 | 95.8% | 93.3% | 0.1020 |
| 50000 | 95.6% | 93.0% | 0.1019 |

---

## stage1_kv_proj_interleaved_v3

**Dataset:** v3 | **LR:** 1e-4 const | **Steps:** 50k

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 39.4% | 63.7% | 0.2875 |
| 10000 | 71.1% | 76.0% | 0.2147 |
| 15000 | 77.9% | 79.7% | 0.1908 |
| 20000 | 87.1% | 82.7% | 0.1824 |
| 25000 | 90.0% | 85.7% | 0.1648 |
| 30000 | 92.3% | 87.7% | 0.1630 |
| 35000 | 92.9% | 84.3% | 0.1572 |
| 40000 | 94.1% | 89.7% | 0.1532 |
| 45000 | 94.4% | 91.7% | 0.1535 |
| 50000 | 95.0% | 91.3% | 0.1583 |

---

## stage1_kv_proj_interleaved_v3_hilr

**Dataset:** v3 | **LR:** 2e-4 cosine | **Steps:** 50k

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 79.8% | 73.3% | 0.1489 |
| 10000 | 89.9% | 81.7% | 0.1240 |
| 15000 | 93.1% | 87.0% | 0.1126 |
| 20000 | 96.3% | 90.3% | 0.1144 |
| 25000 | 96.1% | 93.0% | 0.1030 |
| 30000 | 97.1% | 94.7% | 0.1055 |
| 35000 | 97.4% | 96.7% | 0.0986 |
| 40000 | 97.9% | 96.3% | 0.0956 |
| 45000 | 97.9% | 97.0% | 0.0966 |
| 50000 | 97.9% | 97.0% | 0.1002 |

---

## stage1_llava_v2.1

**Dataset:** v2.1 | **LR:** 1e-4 const | **Steps:** 50k

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 35.1% | 81.7% | 0.2771 |
| 10000 | 100.0% | 94.0% | 0.1773 |
| 15000 | 100.0% | 98.7% | 0.1575 |
| 20000 | 100.0% | 100.0% | 0.1515 |
| 25000 | 100.0% | 99.0% | 0.1498 |
| 30000 | 100.0% | 99.7% | 0.1491 |
| 35000 | 99.9% | 98.7% | 0.1419 |
| 40000 | 100.0% | 100.0% | 0.1420 |
| 45000 | 99.9% | 98.7% | 0.1447 |
| 50000 | 100.0% | 100.0% | 0.1436 |

Note: sq/correct at step 5k is anomalously low (35%) — LLaVA has an initial phase where it learns sq and pc at different rates. pc races ahead early, sq catches up sharply by step 10k.

---

## stage1_llava_v2.1_hilr

**Dataset:** v2.1 | **LR:** 2e-4 cosine | **Steps:** 35k† (24h wall time)

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 93.1% | 83.0% | 0.1408 |
| 10000 | 99.9% | 96.3% | 0.1036 |
| 15000 | 99.8% | 94.7% | 0.1009 |
| 20000 | 100.0% | 99.3% | 0.0937 |
| 25000 | 100.0% | 100.0% | 0.0952 |
| 30000 | 100.0% | 100.0% | 0.0953 |
| 35000 | 100.0% | 100.0% | 0.0922 |

---

## stage1_llava_v3

**Dataset:** v3 | **LR:** 1e-4 const | **Steps:** 50k

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 24.1% | 58.7% | 0.3764 |
| 10000 | 40.0% | 76.7% | 0.2507 |
| 15000 | 99.3% | 90.3% | 0.2096 |
| 20000 | 100.0% | 94.0% | 0.2090 |
| 25000 | 99.9% | 94.7% | 0.1880 |
| 30000 | 99.4% | 93.0% | 0.1842 |
| 35000 | 97.1% | 89.3% | 0.2016 |
| 40000 | 100.0% | 98.7% | 0.1794 |
| 45000 | 97.1% | 92.3% | 0.2089 |
| 50000 | 99.4% | 97.0% | 0.1758 |

Note: noisy training — sq/correct oscillates between 97–100% after step 15k rather than converging cleanly. The loss also fails to decrease monotonically after step 20k, suggesting the constant lr is too high for v3.

---

## stage1_llava_v3_hilr

**Dataset:** v3 | **LR:** 2e-4 cosine | **Steps:** 35k† (24h wall time)

| step | sq/correct | pc/correct | loss |
|------|-----------|-----------|------|
| 5000 | 94.9% | 73.0% | 0.1739 |
| 10000 | 95.5% | 84.3% | 0.1287 |
| 15000 | 98.1% | 93.7% | 0.1089 |
| 20000 | 99.3% | 96.7% | 0.1054 |
| 25000 | 100.0% | 99.3% | 0.1046 |
| 30000 | 99.3% | 97.0% | 0.0992 |
| 35000 | 100.0% | 100.0% | 0.0980 |
