"""Initialization and collation utilities for FlamingoChessLM Stage 1 training."""
import functools
import json
from pathlib import Path

import chess
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from chesslm.encoder.lc0_hf_bt5.hf_model import Lc0Bt4HFModel
from chesslm.models import FlamingoChessLM, LLaVAChessLM
from chesslm.models.base import unwrap_decoder
from chesslm.utils.utils import (
    ANSWER_SPECIAL_TOKENS,
    POV_ANSWER_SPECIAL_TOKENS,
    EMPTY_TOKEN,
    PIECE_TOKENS,
    POV_SQUARE_TOKENS,
    SQUARE_TOKENS,
    SYSTEM_PROMPT,
    _PIECE_TO_TOKEN,
    encode_positions,
)

_COLOR_WORDS = {chess.WHITE: "white", chess.BLACK: "black"}
_PIECE_WORDS = {
    chess.PAWN:   "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK:   "rook",
    chess.QUEEN:  "queen",
    chess.KING:   "king",
}


# ---------------------------------------------------------------------------
# Embedding initialization
# ---------------------------------------------------------------------------

def _mean_embedding(embed_weight: torch.Tensor, tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return embed_weight[ids].float().mean(dim=0)


def init_special_token_embeddings(
    model,  # any ChessLM-conforming arch (Flamingo / LLaVA)
    tokenizer,
    strategy: str,
    pov: bool = False,
) -> None:
    """Initialize model.new_embed and model.new_lm_head weights.

    Reads from the frozen pretrained embed_tokens to compute averages — the
    pretrained weights themselves are never modified.

    strategy='semantic':
      Board-absolute (pov=False):
        SQUARE_XY tokens ← mean(file_char, rank_char) embeddings
      POV-relative (pov=True):
        SQUARE_N tokens ← copy of the corresponding SQUARE_XY semantic init
        (POV index i == board square i for white, so <SQUARE_1> starts as
        a synonym for <SQUARE_A1>; the model shifts during training)
      Piece tokens ← mean(color_word, piece_word) embeddings  (both variants)
      EMPTY token  ← embedding of 'empty'                     (both variants)
    strategy='random': no-op; keeps default random init.
    """
    if strategy == "random" or model.n_new_tokens == 0:
        return

    # Unwrap once and read everything from the same underlying HF model — keeps
    # `frozen_w` and `frozen_vocab` invariant to PEFT proxy behavior.
    base_decoder = unwrap_decoder(model.decoder)
    frozen_w     = base_decoder.model.embed_tokens.weight.data
    new_emb_w    = model.new_embed.weight.data
    frozen_vocab = base_decoder.config.vocab_size

    # Pre-compute semantic embedding for each board-absolute square
    sq_semantic = {}
    for sq in chess.SQUARES:
        sq_name = chess.square_name(sq)
        sq_semantic[sq] = (
            (_mean_embedding(frozen_w, tokenizer, sq_name[0])
           + _mean_embedding(frozen_w, tokenizer, sq_name[1])) / 2.0
        )

    if pov:
        # POV index i == board square i (white-POV correspondence)
        for i, tok in enumerate(POV_SQUARE_TOKENS):
            idx = tokenizer.convert_tokens_to_ids(tok) - frozen_vocab
            new_emb_w[idx] = sq_semantic[i].to(new_emb_w.dtype)
    else:
        for sq in chess.SQUARES:
            idx = tokenizer.convert_tokens_to_ids(SQUARE_TOKENS[sq]) - frozen_vocab
            new_emb_w[idx] = sq_semantic[sq].to(new_emb_w.dtype)

    for (color, ptype), tok in _PIECE_TO_TOKEN.items():
        avg = (_mean_embedding(frozen_w, tokenizer, _COLOR_WORDS[color])
             + _mean_embedding(frozen_w, tokenizer, _PIECE_WORDS[ptype])) / 2.0
        idx = tokenizer.convert_tokens_to_ids(tok) - frozen_vocab
        new_emb_w[idx] = avg.to(new_emb_w.dtype)

    empty_idx = tokenizer.convert_tokens_to_ids(EMPTY_TOKEN) - frozen_vocab
    new_emb_w[empty_idx] = _mean_embedding(frozen_w, tokenizer, "empty").to(new_emb_w.dtype)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(
    batch: list[dict],
    *,
    tokenizer,
    system_prompt: str,
    max_seq_len: int,
) -> dict:
    """Tokenize + right-pad a batch to the longest sequence in the batch.

    Labels are -100 on the prompt (system + user + assistant prefix) and
    the actual token IDs on the answer (answer text + parse tag + <|im_end|>).
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids_list, labels_list = [], []

    for ex in batch:
        full_msgs = [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ]
        prompt_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": ex["question"]},
        ]

        full_ids   = tokenizer.apply_chat_template(full_msgs,   tokenize=True, add_generation_prompt=False)
        prompt_ids = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)

        # Safety truncation — preserves prompt; silently drops tail of very long answers
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels     = [-100] * prompt_len + full_ids[prompt_len:]

        input_ids_list.append(full_ids)
        labels_list.append(labels)

    max_len = max(len(ids) for ids in input_ids_list)
    B = len(batch)

    input_ids  = torch.full((B, max_len), pad_id,  dtype=torch.long)
    attn_mask  = torch.zeros(B, max_len,            dtype=torch.long)
    labels_out = torch.full((B, max_len), -100,     dtype=torch.long)

    for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
        L = len(ids)
        input_ids [i, :L] = torch.tensor(ids,  dtype=torch.long)
        attn_mask [i, :L] = 1
        labels_out[i, :L] = torch.tensor(labs, dtype=torch.long)

    return {
        "input_ids":      input_ids,
        "attention_mask": attn_mask,
        "labels":         labels_out,
        "start_fens":     [x["start_fen"]    for x in batch],
        "moves":          [x["moves"]         for x in batch],
        "fens":           [x["fen"]           for x in batch],
        "question_types": [x["question_type"] for x in batch],
        "answer_classes": [x["answer_class"]  for x in batch],
    }


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def init_model_and_tokenizer(args, special_tokens: list[str] = None, pov: bool = False):
    """Load FlamingoChessLM + LC0 encoder; tokenizer extended with special tokens.

    Pretrained decoder weights are never modified. New token embeddings live in
    model.new_embed / model.new_lm_head (separate trainable modules).
    """
    if special_tokens is None:
        special_tokens = ANSWER_SPECIAL_TOKENS

    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_path, local_files_only=True)
    orig_vocab_size = len(tokenizer)
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    n_new_tokens = len(tokenizer) - orig_vocab_size
    assert n_new_tokens == len(special_tokens), (
        f"Expected {len(special_tokens)} new tokens but got {n_new_tokens}; "
        "some special tokens may already exist in the tokenizer vocab"
    )

    arch = getattr(args, "arch", "flamingo")
    print(f"Loading model (arch={arch})...")
    if arch == "flamingo":
        x_attn_kwargs = {
            "alpha_init":   getattr(args, "alpha_init",   0.0),
            "wo_zero_init": getattr(args, "wo_zero_init", False),
        }
        model = FlamingoChessLM.from_pretrained(
            args.decoder_path,
            n_new_tokens=n_new_tokens,
            lora_rank=getattr(args, "lora_rank", -1),
            x_attn_kwargs=x_attn_kwargs,
            device=device,
            torch_dtype=amp_dtype,
            local_files_only=True,
        )
    elif arch == "llava":
        model = LLaVAChessLM.from_pretrained(
            args.decoder_path,
            n_new_tokens=n_new_tokens,
            lora_rank=getattr(args, "lora_rank", 0),
            device=device,
            torch_dtype=amp_dtype,
            local_files_only=True,
        )
    else:
        raise NotImplementedError(f"arch={arch!r} not yet implemented")

    assert orig_vocab_size == model.decoder.config.vocab_size, (
        f"Tokenizer vocab size ({orig_vocab_size}) != decoder config.vocab_size "
        f"({model.decoder.config.vocab_size}); frozen_vocab boundary would be wrong"
    )

    print("Loading LC0 encoder...")
    encoder = Lc0Bt4HFModel.from_pretrained(args.encoder_path, local_files_only=True)
    encoder.to(device=device, dtype=amp_dtype).eval()

    init_special_token_embeddings(model, tokenizer, args.embed_init, pov=pov)

    model.train()

    if arch == "flamingo":
        # Alpha gates must stay fp32 even in a bf16 run. bf16 precision near 0.55
        # is ~0.004 but per-step alpha updates are ~1e-6, so every update would
        # round to zero and the parameter would never move.
        for layer in model.x_attn_layers:
            layer.alpha_attn.data = layer.alpha_attn.data.float()
            layer.alpha_ffn.data  = layer.alpha_ffn.data.float()
        model.x_attn_layers.train()

    # Keep decoder in eval mode when it is fully frozen (lora_rank < 0).
    # With LoRA (lora_rank > 0) the backbone is frozen but adapters need train
    # mode for lora_dropout; with lora_rank == 0 the whole decoder trains.
    if model.lora_rank < 0:
        model.decoder.eval()

    return model, encoder, tokenizer


def _load_dataset_config(dataset_dir: str) -> dict:
    """Read dataset_config.json if present; return defaults otherwise."""
    cfg_path = Path(dataset_dir).parent / "dataset_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {"pov": False, "new_tok_in_query": False}


def init_datasets_and_dataloader(args, tokenizer):
    """Load HF datasets; return (train DataLoader, eval dataset, dataset config)."""
    # dataset_config.json lives one level up from train/ and eval/
    cfg = _load_dataset_config(args.train_dataset)
    print(f"Dataset config: {cfg}")

    print(f"Loading train dataset from {args.train_dataset}...")
    train_ds = load_from_disk(args.train_dataset)
    print(f"  train: {len(train_ds)} examples")
    print(f"Loading eval dataset from {args.eval_dataset}...")
    eval_ds  = load_from_disk(args.eval_dataset)
    print(f"  eval:  {len(eval_ds)} examples")

    cfn = functools.partial(
        collate_fn,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        max_seq_len=args.max_seq_len,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=cfn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, eval_ds, cfg


def post_eval(args, model) -> None:
    """Restore train mode after eval. Mirrors init_model_and_tokenizer's
    train/eval setup so eval cycles don't leave the model in a wrong mode."""
    model.train()
    raw = getattr(model, "_orig_mod", model)
    if getattr(args, "arch", "flamingo") == "flamingo":
        raw.x_attn_layers.train()
    # Keep decoder in eval when fully frozen; LoRA / full-train need train mode.
    if raw.lora_rank < 0:
        raw.decoder.eval()


def init_optimizer_and_scheduler(args, model):
    decoder_lr = getattr(args, "decoder_lr", None)
    optimizer = torch.optim.AdamW(model.param_groups(args.lr, decoder_lr=decoder_lr), weight_decay=args.weight_decay)
    warmup_steps = int(getattr(args, "warmup_ratio", 0.0) * args.n_steps)
    if args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.n_steps)
    elif args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, args.n_steps)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    return optimizer, scheduler


def initialize_training_objects(args):
    """Top-level init. Returns everything needed by the training loop."""
    # Load dataset config first so the correct token set is added to the tokenizer.
    cfg = _load_dataset_config(args.train_dataset)
    special_tokens = POV_ANSWER_SPECIAL_TOKENS if cfg.get("pov") else ANSWER_SPECIAL_TOKENS

    model, encoder, tokenizer  = init_model_and_tokenizer(args, special_tokens, pov=cfg.get("pov", False))
    train_loader, eval_ds, _   = init_datasets_and_dataloader(args, tokenizer)
    optimizer, scheduler       = init_optimizer_and_scheduler(args, model)

    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.dtype]
    special_token_ids = {tokenizer.convert_tokens_to_ids(t) for t in special_tokens}
    id_to_special     = {tokenizer.convert_tokens_to_ids(t): t for t in special_tokens}

    return (
        model, encoder, tokenizer,
        train_loader, eval_ds,
        optimizer, scheduler,
        amp_dtype, special_token_ids, id_to_special,
        cfg,
    )
