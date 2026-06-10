import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from itertools import chain, repeat

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin

from chesslm.utils.training_utils import initialize_training_objects, post_eval
from chesslm.utils.eval_utils import run_eval
from chesslm.utils.utils import encode_positions


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(accelerator, run_dir: Path, step: int) -> Path:
    # accelerator.save_state writes the FSDP-sharded model + optimizer +
    # scheduler + RNG; the step goes in a small sidecar (Step 4 adds the data
    # sampler state here too).
    ckpt_dir = run_dir / f"step_{step:07d}"
    accelerator.save_state(str(ckpt_dir))
    if accelerator.is_main_process:
        (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step}))
        print(f"[step {step}] checkpoint saved → {ckpt_dir}")
    return ckpt_dir


def load_checkpoint(accelerator, ckpt_dir: str) -> int:
    accelerator.load_state(str(ckpt_dir))
    return json.loads((Path(ckpt_dir) / "train_state.json").read_text())["step"]


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
    g.add_argument("--arch",      choices=["flamingo", "llava"], default="flamingo")
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

    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.dtype]

    # --- accelerate + FSDP2 ---
    # Wrap each decoder layer and cross-attention bridge as its own unit (the
    # models' `_no_split_modules`). FSDP2 shards each parameter independently,
    # so the bf16 weights and the fp32 flamingo alpha gates keep their dtypes.
    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["SmolLM3DecoderLayer", "DenseXAttn"],
        reshard_after_forward=True,
        state_dict_type="SHARDED_STATE_DICT",
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        fsdp_plugin=fsdp_plugin,
    )
    device = accelerator.device
    args.device = str(device)  # build the model + encoder on this rank's GPU

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = Path(args.output_dir) / args.exp_name
    if accelerator.is_main_process:
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

    # Shard the model + optimizer + scheduler + dataloader across ranks. The
    # frozen encoder is not trained, so it stays replicated (not prepared).
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler,
    )
    accelerator.print(f"Initialization complete ({time.time() - t0:.1f}s)")

    if args.compile:
        accelerator.print("Compiling model with torch.compile...")
        t0 = time.time()
        model = torch.compile(model, mode="reduce-overhead")
        accelerator.print(f"Compilation done ({time.time() - t0:.1f}s)")

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(accelerator, args.resume_from)
        accelerator.print(f"Resumed from step {start_step}")

    train_iter = chain.from_iterable(repeat(train_loader))

    def do_eval(step: int, jsonl_file, txt_file, train_loss: float | None = None) -> None:
        # Runs on every rank — generation forwards trigger FSDP all-gathers, so
        # all ranks must participate — but only the main process logs / saves.
        accelerator.print(f"[eval] starting eval at step {step}...")
        t_eval = time.time()
        model.eval()
        metrics, samples = run_eval(
            model, encoder, eval_ds, tokenizer, device, amp_dtype,
            args.eval_batch_size, args.eval_max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            max_examples=args.eval_max_examples,
            encode_pov=encode_pov,
        )
        diag = get_diagnostics(accelerator.unwrap_model(model))
        full_metrics = ({"train_loss": train_loss} if train_loss is not None else {}) | metrics | diag
        if accelerator.is_main_process:
            step_dir = run_dir / f"step_{step:07d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            save_generations(step_dir, samples)
            log_metrics(step, full_metrics, jsonl_file, txt_file)
            if verbose and args.log_samples > 0:
                for s in samples[:args.log_samples]:
                    print(f"  [{s['qt']}] {s['question'][:60]} → {s['generated'][:80]}")
        accelerator.print(f"[eval] done ({time.time() - t_eval:.1f}s) — {len(samples)} generations")
        post_eval(args, accelerator.unwrap_model(model))

    # Metrics files are written by the main process only.
    jsonl_file = open(jsonl_path, "a") if accelerator.is_main_process else None
    txt_file   = open(txt_path, "a") if accelerator.is_main_process else None

    if args.eval_at_start:
        do_eval(start_step, jsonl_file, txt_file)

    optimizer.zero_grad()
    accelerator.print(
        f"Training started — {args.n_steps} steps, batch={args.batch_size}, "
        f"grad_accum={args.grad_accum_steps}, world_size={accelerator.num_processes}, "
        f"effective_batch={args.batch_size * args.grad_accum_steps * accelerator.num_processes}"
    )

    step_iter = range(start_step, args.n_steps)
    if verbose and accelerator.is_main_process:
        from tqdm import tqdm
        step_iter = tqdm(step_iter, desc="train", dynamic_ncols=True)

    for step in step_iter:

        # --- gradient accumulation (accelerate gates grad sync + optimizer step) ---
        accum_loss = 0.0
        for _ in range(args.grad_accum_steps):
            batch = next(train_iter)
            with accelerator.accumulate(model):
                enc_hidden = encode_positions(
                    encoder, batch["start_fens"], batch["moves"], batch["fens"],
                    device, amp_dtype, pov=encode_pov,
                )
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    loss = model(batch["input_ids"], enc_hidden,
                                 batch["attention_mask"], labels=batch["labels"])
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_loss += loss.item() / args.grad_accum_steps

        if verbose and accelerator.is_main_process:
            step_iter.set_postfix(loss=f"{accum_loss:.4f}")
        elif (step + 1) % 50 == 0:
            accelerator.print(f"[step {step + 1}/{args.n_steps}] loss={accum_loss:.4f}")

        # --- eval + checkpoint ---
        if (step + 1) % args.eval_freq == 0:
            do_eval(step + 1, jsonl_file, txt_file, train_loss=accum_loss)
            save_checkpoint(accelerator, run_dir, step + 1)

    if accelerator.is_main_process:
        if jsonl_file is not None:
            jsonl_file.close()
        if txt_file is not None:
            txt_file.close()
    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
