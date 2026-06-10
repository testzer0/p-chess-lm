"""Generative evaluation for FlamingoChessLM Stage 1 training."""
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chesslm.utils.utils import (
    ANSWER_SPECIAL_TOKENS,
    POV_ANSWER_SPECIAL_TOKENS,
    EMPTY_TOKEN,
    PIECE_TOKENS,
    POV_SQUARE_TOKENS,
    SQUARE_TOKENS,
    SYSTEM_PROMPT,
    _generate_position_dict,
    _generate_pov_position_dict,
    encode_positions,
)

_SQUARE_SET     = set(SQUARE_TOKENS)
_POV_SQUARE_SET = set(POV_SQUARE_TOKENS)
_PIECE_SET      = set(PIECE_TOKENS)


# ---------------------------------------------------------------------------
# Parse-tag validation
# ---------------------------------------------------------------------------

def _is_valid_parse_tag(toks: list[str], question_type: str) -> bool:
    """True iff toks form a well-structured parse tag for question_type."""
    if not toks:
        return False

    if question_type in ("static_square", "static_square_pov"):
        sq_set = _SQUARE_SET if question_type == "static_square" else _POV_SQUARE_SET
        return (len(toks) == 2
                and toks[0] in sq_set
                and toks[1] in _PIECE_SET | {EMPTY_TOKEN})

    if question_type in ("static_piece", "static_piece_pov"):
        sq_set = _SQUARE_SET if question_type == "static_piece" else _POV_SQUARE_SET
        if toks[0] not in _PIECE_SET:
            return False
        rest = toks[1:]
        if not rest:
            return False
        if rest == [EMPTY_TOKEN]:
            return True
        return all(t in sq_set for t in rest)

    return False


def _is_consistent(question_type: str, toks: list[str], fen: str) -> bool:
    """True iff the parse tag is consistent with the board position."""

    if question_type == "static_square":
        if len(toks) != 2:
            return False
        sq_tok, piece_tok = toks
        if sq_tok not in _SQUARE_SET:
            return False
        sq2p, _ = _generate_position_dict(fen)
        sq_name = sq_tok[8:-1].lower()   # "<SQUARE_E4>" → "e4"
        return sq2p.get(sq_name) == piece_tok

    if question_type == "static_piece":
        if not toks or toks[0] not in _PIECE_SET:
            return False
        _, p2sq = _generate_position_dict(fen)
        piece_tok, rest = toks[0], toks[1:]
        actual_sqs = p2sq.get(piece_tok, [])
        if not actual_sqs:
            return rest == [EMPTY_TOKEN]
        if any(t not in _SQUARE_SET for t in rest):
            return False
        return sorted(t[8:-1].lower() for t in rest) == sorted(actual_sqs)

    if question_type == "static_square_pov":
        if len(toks) != 2:
            return False
        sq_tok, piece_tok = toks
        if sq_tok not in _POV_SQUARE_SET:
            return False
        pov_idx = int(sq_tok[8:-1]) - 1          # "<SQUARE_7>" → 6
        pov2p, _ = _generate_pov_position_dict(fen)
        return pov2p.get(pov_idx) == piece_tok

    if question_type == "static_piece_pov":
        if not toks or toks[0] not in _PIECE_SET:
            return False
        _, p2idxs = _generate_pov_position_dict(fen)
        piece_tok, rest = toks[0], toks[1:]
        actual_idxs = p2idxs.get(piece_tok, [])
        if not actual_idxs:
            return rest == [EMPTY_TOKEN]
        if any(t not in _POV_SQUARE_SET for t in rest):
            return False
        pred_idxs = sorted(int(t[8:-1]) - 1 for t in rest)
        return pred_idxs == sorted(actual_idxs)

    return False


# ---------------------------------------------------------------------------
# Batched greedy generation
# ---------------------------------------------------------------------------

def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Apply temperature / top-k / top-p and return next token ids (B,).

    temperature=0 → greedy argmax (no sampling).
    Mirrors the order transformers uses: temperature → top-k → top-p → multinomial.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    scores = logits / temperature

    if top_k > 0:
        k = min(top_k, scores.size(-1))
        cutoff = torch.topk(scores, k)[0][..., -1, None]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scores, descending=False)
        cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumprobs <= (1 - top_p)
        remove[..., -1:] = False  # always keep at least one token
        scores = scores.masked_fill(
            remove.scatter(1, sorted_idx, remove), float("-inf")
        )

    return torch.multinomial(scores.softmax(dim=-1), num_samples=1).squeeze(-1)


def _batched_decode(
    model,
    enc_hidden: torch.Tensor,
    prompt_ids_list: list[list[int]],
    pad_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    temperature: float = 0.0,
    top_k: int = 20,
    top_p: float = 0.95,
) -> list[list[int]]:
    """Decode a batch of prompts, stopping each example at EOS.

    Returns a list of generated token ID lists (prompt tokens excluded).
    temperature=0 uses greedy argmax; >0 uses top-k/top-p sampling.

    Left-padding: padding is prepended so all real tokens end at position S-1.
    position_ids = attn_mask.cumsum(-1) - 1 gives each token its correct RoPE
    position (pads get 0, which is masked anyway). Generated tokens append to
    the right and always get the next correct position index.
    """
    B = len(prompt_ids_list)
    prompt_lens = [len(ids) for ids in prompt_ids_list]
    max_prompt  = max(prompt_lens)

    input_ids = torch.full((B, max_prompt), pad_token_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros(B, max_prompt,                dtype=torch.long, device=device)
    for i, (ids, l) in enumerate(zip(prompt_ids_list, prompt_lens)):
        input_ids[i, max_prompt - l:] = torch.tensor(ids, dtype=torch.long, device=device)
        attn_mask[i, max_prompt - l:] = 1

    done      = torch.zeros(B, dtype=torch.bool, device=device)
    generated = [[] for _ in range(B)]

    for _ in range(max_new_tokens):
        if done.all():
            break

        position_ids = (attn_mask.cumsum(dim=-1) - 1).clamp(min=0)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            logits = model(input_ids, enc_hidden, attn_mask, position_ids=position_ids)

        next_logits = logits[:, -1, :]                                            # (B, V)
        next_tokens = _sample_next_token(next_logits, temperature, top_k, top_p)  # (B,)

        new_col   = torch.where(done, torch.full_like(next_tokens, pad_token_id), next_tokens)
        input_ids = torch.cat([input_ids, new_col.unsqueeze(1)], dim=1)
        attn_mask = torch.cat([attn_mask, (~done).long().unsqueeze(1)], dim=1)

        for i in range(B):
            if not done[i]:
                tok = next_tokens[i].item()
                generated[i].append(tok)
                if tok == eos_token_id:
                    done[i] = True

    return generated


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def _eval_collate(batch: list[dict]) -> dict:
    return {k: [ex[k] for ex in batch] for k in batch[0]}


@torch.no_grad()
def run_eval(
    model,
    encoder,
    eval_dataset,
    tokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    eval_batch_size: int,
    eval_max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 20,
    top_p: float = 0.95,
    max_examples: int | None = None,
    encode_pov: bool = False,
) -> tuple[dict[str, float], list[dict]]:
    """Run generative eval over the eval dataset (or a capped subset).

    Returns (metrics, samples) where samples contains every evaluated example.
    """
    model.eval()

    if max_examples is not None:
        eval_dataset = eval_dataset.select(range(min(max_examples, len(eval_dataset))))

    _special_tokens = POV_ANSWER_SPECIAL_TOKENS if encode_pov else ANSWER_SPECIAL_TOKENS
    id_to_special = {tokenizer.convert_tokens_to_ids(t): t for t in _special_tokens}
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None, "tokenizer.eos_token_id is None — generation cannot detect stop"
    # Prefer pad_token_id; fall back to eos but warn so the user can confirm the
    # collator/decoder didn't silently switch padding semantics mid-run.
    if tokenizer.pad_token_id is None:
        print(f"[run_eval] WARNING: tokenizer.pad_token_id is None; using eos_token_id ({eos_id}) as pad.")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=_eval_collate,
    )

    stats   = defaultdict(lambda: {"total": 0, "parse_valid": 0, "correct": 0, "consistent": 0})
    samples = []

    if os.environ.get("TEST_MODE") == "1":
        from tqdm import tqdm
        loader = tqdm(loader, desc="eval batches", dynamic_ncols=True)

    for batch in loader:
        # Build prompt-only token sequences (no answer)
        prompt_ids_list = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user",   "content": q}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for q in batch["question"]
        ]

        enc_hidden = encode_positions(
            encoder,
            batch["start_fen"],
            batch["moves"],
            batch["fen"],
            device,
            amp_dtype,
            pov=encode_pov,
        )

        gen_ids_list = _batched_decode(
            model, enc_hidden, prompt_ids_list,
            pad_id, eos_id, eval_max_new_tokens, device, amp_dtype,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )

        for i, gen_ids in enumerate(gen_ids_list):
            qt        = batch["question_type"][i]
            fen       = batch["fen"][i]
            gt_class  = batch["answer_class"][i]
            question  = batch["question"][i]

            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            parse_region   = generated_text.rsplit('\n\n', 1)[-1]
            parse_ids      = tokenizer.encode(parse_region, add_special_tokens=False)
            pred_toks      = [id_to_special[t] for t in parse_ids if t in id_to_special]
            valid       = _is_valid_parse_tag(pred_toks, qt)
            correct     = valid and (pred_toks == gt_class)
            consistent  = valid and _is_consistent(qt, pred_toks, fen)

            stats[qt]["total"]      += 1
            stats[qt]["parse_valid"] += int(valid)
            stats[qt]["correct"]     += int(correct)
            stats[qt]["consistent"]  += int(consistent)
            samples.append({
                "qt":        qt,
                "fen":       fen,
                "question":  question,
                "generated": generated_text,
                "gt_class":  gt_class,
                "pred_toks": pred_toks,
                "correct":   correct,
            })

    metrics: dict[str, float] = {}
    for qt, s in stats.items():
        n = s["total"]
        metrics[f"{qt}/parse_valid"]   = s["parse_valid"]  / n if n else 0.0
        metrics[f"{qt}/correct"]       = s["correct"]      / n if n else 0.0
        metrics[f"{qt}/fen_consistent"] = s["consistent"]  / n if n else 0.0

    return metrics, samples
