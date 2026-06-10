import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from itertools import chain, repeat

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from chesslm.utils.training_utils import initialize_training_objects, post_eval
from chesslm.utils.eval_utils import run_eval
from chesslm.utils.utils import encode_positions


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def ntp_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shift logits/labels by 1, then NTP cross-entropy ignoring -100."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(run_dir: Path, step: int, model, optimizer, scheduler) -> None:
    ckpt_dir = run_dir / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":      step,
        "model":     model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, ckpt_dir / "checkpoint.pt")
    print(f"[step {step}] checkpoint saved → {ckpt_dir}")
    return ckpt_dir


def load_checkpoint(ckpt_path: str, model, optimizer, scheduler) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_trainable_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_metrics(step: int, metrics: dict, jsonl_file, txt_file) -> None:
    row = {"step": step, **metrics}
    jsonl_file.write(json.dumps(row) + "\n")
    jsonl_file.flush()

    txt_file.write(f"step={step}\n")
    for k, v in metrics.items():
        txt_file.write(f"  {k:<40s} {v:.4f}\n")
    txt_file.write("\n")
    txt_file.flush()

    print(f"[step {step}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


def save_generations(ckpt_dir: Path, samples: list[dict]) -> None:
    with open(ckpt_dir / "generations.json", "w") as f:
        json.dump(samples, f, indent=2)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train FlamingoChessLM Stage 1")

    # --- reproducibility ---
    g = parser.add_argument_group("reproducibility")
    g.add_argument("--seed", type=int, default=42)

    # --- training ---
    g = parser.add_argument_group("training")
    g.add_argument("--batch-size",       type=int,   default=256)
    g.add_argument("--max-seq-len",      type=int,   default=256)
    g.add_argument("--n-steps",          type=int,   default=10_000)
    g.add_argument("--lr",               type=float, default=1e-4)
    g.add_argument("--decoder-lr",       type=float, default=None,
                   help="LR for decoder LoRA group; defaults to --lr if not set")
    g.add_argument("--grad-accum-steps", type=int,   default=1)
    g.add_argument("--max-grad-norm",    type=float, default=1.0)

    # --- optimizer / scheduler ---
    g = parser.add_argument_group("optimizer")
    g.add_argument("--weight-decay",  type=float, default=0.01)
    g.add_argument("--embed-init",    choices=["semantic", "random"], default="semantic")
    g.add_argument("--scheduler",     choices=["constant", "cosine", "linear"], default="constant")
    g.add_argument("--warmup-ratio",  type=float, default=0.05,
                   help="Fraction of n_steps used for linear warmup (cosine/linear schedulers only)")

    # --- eval ---
    g = parser.add_argument_group("eval")
    g.add_argument("--eval-freq",           type=int,   default=500)
    g.add_argument("--eval-batch-size",     type=int,   default=64)
    g.add_argument("--eval-max-new-tokens", type=int,   default=128)
    g.add_argument("--eval-max-examples",   type=int,   default=1900,
                   help="Cap eval dataset size (default 1900 = 25 positions × 76 examples)")
    g.add_argument("--log-samples",         type=int,   default=4)
    g.add_argument("--eval-at-start",       action="store_true", default=False)
    g.add_argument("--temperature",         type=float, default=0.0,
                   help="Sampling temperature; 0 = greedy")
    g.add_argument("--top-k",               type=int,   default=20)
    g.add_argument("--top-p",               type=float, default=0.95)

    # --- model ---
    g = parser.add_argument_group("model")
    g.add_argument("--arch",      choices=["flamingo", "llava", "kv_proj"], default="flamingo")
    g.add_argument("--proj-mode", choices=["channel_concat", "interleaved"], default="channel_concat",
                   help="KV projection mode (kv_proj arch only)")
    g.add_argument("--lora-rank", type=int, default=-1,
                   help="LoRA rank: <0 = frozen decoder, 0 = full fine-tuning, >0 = LoRA adapters")
    g.add_argument("--decoder-path", required=True)
    g.add_argument("--encoder-path", required=True)
    g.add_argument("--alpha-init",    type=float, default=0.0,
                   help="Initial alpha value (pre-tanh). 0.0=Flamingo original (default), 0.5493=atanh(0.5)")
    g.add_argument("--wo-zero-init",  action="store_true",
                   help="Zero-initialize W_O (default: random init).")
    g.add_argument("--dtype",        choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    g.add_argument("--device",       default="cuda")
    g.add_argument("--compile",      action="store_true", default=False,
                   help="torch.compile the model (faster steady-state, ~2min cold-start)")

    # --- data ---
    g = parser.add_argument_group("data")
    g.add_argument("--train-dataset", required=True, help="Path to HF Arrow train dataset dir")
    g.add_argument("--eval-dataset",  required=True, help="Path to HF Arrow eval dataset dir")
    g.add_argument("--num-workers",   type=int, default=4)

    # --- output ---
    g = parser.add_argument_group("output")
    g.add_argument("--exp-name",    required=True, help="Experiment name; outputs go to runs/{exp_name}/")
    g.add_argument("--output-dir",  default="chesslm/runs/")
    g.add_argument("--resume-from", default=None, help="Path to checkpoint.pt to resume from")

    return parser.parse_args()


def get_diagnostics(model) -> dict[str, float]:
    """Unwraps torch.compile's OptimizedModule wrapper before delegating."""
    raw = getattr(model, "_orig_mod", model)
    return raw.get_diagnostics()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    verbose = os.environ.get("TEST_MODE") == "1"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.dtype]

    run_dir = Path(args.output_dir) / args.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "metrics.jsonl"
    txt_path   = run_dir / "metrics.txt"

    t0 = time.time()
    (model, encoder, tokenizer,
     train_loader, eval_ds,
     optimizer, scheduler,
     amp_dtype, special_token_ids, id_to_special,
     dataset_cfg) = initialize_training_objects(args)
    encode_pov = dataset_cfg.get("pov", False)
    print(f"Initialization complete ({time.time() - t0:.1f}s)")

    if args.compile:
        print("Compiling model with torch.compile...")
        t0 = time.time()
        model = torch.compile(model, mode="reduce-overhead")
        print(f"Compilation done ({time.time() - t0:.1f}s)")

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(args.resume_from, model, optimizer, scheduler)
        print(f"Resumed from step {start_step}")

    scaler = torch.amp.GradScaler(device=device.type, enabled=(amp_dtype == torch.float16))
    train_iter = chain.from_iterable(repeat(train_loader))

    def do_eval(step: int, jsonl_file, txt_file, train_loss: float | None = None) -> None:
        print(f"[eval] starting eval at step {step}...")
        t_eval = time.time()
        model.eval()
        metrics, samples = run_eval(
            model, encoder, eval_ds, tokenizer, device, amp_dtype,
            args.eval_batch_size, args.eval_max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            max_examples=args.eval_max_examples,
            encode_pov=encode_pov,
        )
        full_metrics = ({"train_loss": train_loss} if train_loss is not None else {}) | metrics | get_diagnostics(model)
        step_dir = run_dir / f"step_{step:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        save_generations(step_dir, samples)
        log_metrics(step, full_metrics, jsonl_file, txt_file)
        if verbose and args.log_samples > 0:
            for s in samples[:args.log_samples]:
                print(f"  [{s['qt']}] {s['question'][:60]} → {s['generated'][:80]}")
        print(f"[eval] done ({time.time() - t_eval:.1f}s) — {len(samples)} generations saved")
        post_eval(args, model)

    with open(jsonl_path, "a") as jsonl_file, open(txt_path, "a") as txt_file:

        if args.eval_at_start:
            do_eval(start_step, jsonl_file, txt_file)

        optimizer.zero_grad()
        print(f"Training started — {args.n_steps} steps, "
              f"batch={args.batch_size}, grad_accum={args.grad_accum_steps}, "
              f"effective_batch={args.batch_size * args.grad_accum_steps}")

        if verbose:
            from tqdm import tqdm
            step_iter = tqdm(range(start_step, args.n_steps), desc="train", dynamic_ncols=True)
        else:
            step_iter = range(start_step, args.n_steps)

        for step in step_iter:

            # --- gradient accumulation ---
            accum_loss = 0.0
            for _ in range(args.grad_accum_steps):
                batch = next(train_iter)

                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels    = batch["labels"].to(device)

                enc_hidden = encode_positions(
                    encoder,
                    batch["start_fens"],
                    batch["moves"],
                    batch["fens"],
                    device,
                    amp_dtype,
                    pov=encode_pov,
                )

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(input_ids, enc_hidden, attn_mask)
                    loss   = ntp_loss(logits, labels) / args.grad_accum_steps

                scaler.scale(loss).backward()
                accum_loss += loss.item()

            # --- optimizer step ---
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            if verbose:
                step_iter.set_postfix(loss=f"{accum_loss:.4f}")
            elif (step + 1) % 50 == 0:
                print(f"[step {step + 1}/{args.n_steps}] loss={accum_loss:.4f}")

            # --- eval + checkpoint ---
            if (step + 1) % args.eval_freq == 0:
                do_eval(step + 1, jsonl_file, txt_file, train_loss=accum_loss)
                save_checkpoint(run_dir, step + 1, model, optimizer, scheduler)
                # decoder stays in eval (frozen, no dropout noise)

    print("Training complete.")


if __name__ == "__main__":
    main()
