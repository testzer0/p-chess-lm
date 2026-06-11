"""Initialization and collation utilities for chess-LM training."""
import functools

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
from chesslm.utils.instance_format import (
    KEY_EXTRA,
    KEY_FEN,
    KEY_HISTORY,
    KEY_PROMPT,
    KEY_RESPONSE,
    to_standard_instance,
    tokenize_instance,
)
from chesslm.utils.lc0_planes import encode_fen_batch
from chesslm.utils.special_tokens import (
    maybe_add_special_tokens,
    maybe_init_special_token_embeddings,
)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict], *, tokenizer, max_seq_len: int) -> dict:
    """Collate standardized {fen, history, prompt, response, extra} rows.

    Tokenizes prompt+response with the prompt span masked to -100 and EOS
    appended (instance_format.tokenize_instance), right-pads, and builds the lc0
    input planes from fen + history. ``extra`` is carried for eval only and is
    not forwarded to the model.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    cap = max_seq_len or None
    std = [to_standard_instance(ex) for ex in batch]
    toks = [tokenize_instance(tokenizer, s[KEY_PROMPT], s[KEY_RESPONSE], max_length=cap) for s in std]

    B = len(toks)
    max_len = max(len(ids) for ids, _ in toks)
    input_ids  = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask  = torch.zeros(B, max_len,          dtype=torch.long)
    labels_out = torch.full((B, max_len), -100,   dtype=torch.long)
    for i, (ids, labs) in enumerate(toks):
        L = len(ids)
        input_ids [i, :L] = torch.tensor(ids,  dtype=torch.long)
        attn_mask [i, :L] = 1
        labels_out[i, :L] = torch.tensor(labs, dtype=torch.long)

    fens      = [s[KEY_FEN] for s in std]
    histories = [s[KEY_HISTORY] or None for s in std]
    return {
        "input_ids":      input_ids,
        "attention_mask": attn_mask,
        "labels":         labels_out,
        "planes":         encode_fen_batch(fens, histories),
        "extra":          [s[KEY_EXTRA] for s in std],
    }


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def init_model_and_tokenizer(args):
    """Load the chess-LM (flamingo / llava) + lc0 encoder + tokenizer.

    By default the tokenizer is expected to already contain every token the data
    uses (the repo's chess tokenizers do), so nothing is added (n_new_tokens=0).
    Set ``add_special_tokens: true`` in the config to instead add the answer
    tokens here and train new embeddings for them (see utils/special_tokens.py).
    """
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_path, local_files_only=True)
    orig_vocab = len(tokenizer)
    n_new_tokens = maybe_add_special_tokens(tokenizer, args)  # 0 unless flag set

    arch = getattr(args, "arch", "flamingo")
    print(f"Loading model (arch={arch})...")
    if arch == "flamingo":
        x_attn_kwargs = {
            "alpha_init":   getattr(args, "alpha_init",   0.0),
            "wo_zero_init": getattr(args, "wo_zero_init", False),
        }
        model = FlamingoChessLM.from_pretrained(
            args.decoder_path, n_new_tokens=n_new_tokens,
            lora_rank=getattr(args, "lora_rank", -1),
            x_attn_kwargs=x_attn_kwargs,
            device=device, torch_dtype=amp_dtype, local_files_only=True,
        )
    elif arch == "llava":
        model = LLaVAChessLM.from_pretrained(
            args.decoder_path, n_new_tokens=n_new_tokens,
            lora_rank=getattr(args, "lora_rank", 0),
            device=device, torch_dtype=amp_dtype, local_files_only=True,
        )
    else:
        raise NotImplementedError(f"arch={arch!r} not yet implemented")

    assert orig_vocab == model.decoder.config.vocab_size, (
        f"Tokenizer base size ({orig_vocab}) != decoder vocab "
        f"({model.decoder.config.vocab_size}); the tokenizer must match the model."
    )

    print("Loading LC0 encoder...")
    encoder = Lc0Bt4HFModel.from_pretrained(args.encoder_path, local_files_only=True)
    encoder.to(device=device, dtype=amp_dtype).eval()

    maybe_init_special_token_embeddings(model, tokenizer, args)  # no-op unless flag set

    model.train()
    if arch == "flamingo":
        # Alpha gates stay fp32 even in a bf16 run: per-step updates (~1e-6) would
        # round to zero against bf16's ~0.004 precision near 0.55.
        for layer in model.x_attn_layers:
            layer.alpha_attn.data = layer.alpha_attn.data.float()
            layer.alpha_ffn.data  = layer.alpha_ffn.data.float()
        model.x_attn_layers.train()
    # Keep the decoder in eval mode when fully frozen (lora_rank < 0).
    if model.lora_rank < 0:
        model.decoder.eval()

    return model, encoder, tokenizer


def init_datasets_and_dataloader(args, tokenizer):
    """Load the HF (Arrow) train + eval datasets; return (train DataLoader, eval dataset)."""
    print(f"Loading train dataset from {args.train_dataset}...")
    train_ds = load_from_disk(args.train_dataset)
    print(f"  train: {len(train_ds)} examples")
    print(f"Loading eval dataset from {args.eval_dataset}...")
    eval_ds  = load_from_disk(args.eval_dataset)
    print(f"  eval:  {len(eval_ds)} examples")

    cfn = functools.partial(collate_fn, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=cfn, num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, eval_ds


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
    """Top-level init. Returns everything the training loop needs."""
    model, encoder, tokenizer = init_model_and_tokenizer(args)
    train_loader, eval_ds     = init_datasets_and_dataloader(args, tokenizer)
    optimizer, scheduler      = init_optimizer_and_scheduler(args, model)
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.dtype]
    return model, encoder, tokenizer, train_loader, eval_ds, optimizer, scheduler, amp_dtype
