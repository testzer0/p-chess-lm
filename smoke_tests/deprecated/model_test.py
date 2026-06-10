import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import chess
import torch
import torch.nn.functional as F

from chesslm.models import FlamingoChessLM
from chesslm.encoder.lc0_hf_bt5.hf_model import Lc0Bt4HFModel
from chesslm.utils.utils import ANSWER_SPECIAL_TOKENS

N_NEW = len(ANSWER_SPECIAL_TOKENS)  # 77

DECODER_PATH  = '/scratch/gpfs/DANQIC/jeff/models/smollm-3b-instruct'
ENCODER_PATH  = '/scratch/gpfs/DANQIC/jeff/chesslm/chesslm/encoder/lc0_hf_bt5'
POSITIONS_PATH = '/scratch/gpfs/DANQIC/jeff/chesslm/chesslm/raw_data/positions.jsonl'
N_POSITIONS   = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_positions(encoder: Lc0Bt4HFModel, positions: list, batch_size: int = 10) -> torch.Tensor:
    """
    positions: list of [start_fen, moves, end_fen]
    Returns: (B, 16, 64, 1024) float32 encoder hidden states, canonicalized.
    """
    dev = next(encoder.buffers()).device
    black_idx = torch.tensor([sq ^ 56 for sq in range(64)], device=dev)

    all_planes, is_black = [], []
    for start_fen, moves, end_fen in positions:
        all_planes.append(encoder.input_planes_from_fen(start_fen, moves))
        is_black.append(chess.Board(end_fen).turn == chess.BLACK)

    results = []
    for i in range(0, len(all_planes), batch_size):
        batch_planes = torch.stack(all_planes[i:i+batch_size]).to(dev)  # (B, 112, 64)
        out = encoder(batch_planes, output_hidden_states=True)
        # all_hidden_states: tuple of 16 tensors each (B, 64, 1024)
        layer_hidden = torch.stack(out.all_hidden_states, dim=1)  # (B, 16, 64, 1024)

        # Canonicalize per sample
        for j, black in enumerate(is_black[i:i+batch_size]):
            h = layer_hidden[j]  # (16, 64, 1024)
            if black:
                h = h[:, black_idx, :]
            results.append(h)

    return torch.stack(results, dim=0)  # (B, 16, 64, 1024)


def _open_gates(model: FlamingoChessLM, value: float) -> None:
    for layer in model.x_attn_layers:
        layer.alpha_attn.data.fill_(value)
        layer.alpha_ffn.data.fill_(value)


def _zero_gates(model: FlamingoChessLM) -> None:
    _open_gates(model, 0.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def basic_impl_test(model: FlamingoChessLM) -> None:
    print("=== basic_impl_test ===")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.trainable_parameters())
    frozen    = total - trainable
    print(f"  Total params:     {total / 1e6:.1f}M")
    print(f"  Trainable params: {trainable / 1e6:.1f}M")
    print(f"  Frozen params:    {frozen / 1e6:.1f}M")

    B, S = 2, 32
    dev = next(model.parameters()).device
    input_ids  = torch.randint(0, 1000, (B, S), device=dev)
    enc_hidden = torch.randn(B, 16, 64, 1024, dtype=torch.bfloat16, device=dev)

    with torch.no_grad():
        logits = model(input_ids, enc_hidden)

    expected_shape = (B, S, model.decoder.config.vocab_size + model.n_new_tokens)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {tuple(logits.shape)}"
    print(f"  Logits shape: {tuple(logits.shape)}  OK")


def cpu_gpu_decoder_test() -> None:
    """Check whether plain SmolLM3 gives identical logits on CPU vs GPU."""
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available():
        print("=== cpu_gpu_decoder_test: SKIPPED (no GPU) ===")
        return
    print("=== cpu_gpu_decoder_test ===")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DECODER_PATH, local_files_only=True)
    input_ids = tokenizer("this is a test string", return_tensors="pt").input_ids.repeat(2, 1)
    print(f"  input_ids shape: {tuple(input_ids.shape)}")

    cpu_model = AutoModelForCausalLM.from_pretrained(DECODER_PATH, torch_dtype=torch.bfloat16, local_files_only=True)
    cpu_model.eval()
    with torch.no_grad():
        cpu_logits = cpu_model(input_ids).logits

    gpu_model = AutoModelForCausalLM.from_pretrained(DECODER_PATH, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    gpu_model.eval()
    with torch.no_grad():
        gpu_logits = gpu_model(input_ids.cuda()).logits.cpu()

    max_diff = (cpu_logits - gpu_logits).abs().max().item()
    print(f"  CPU vs GPU max logit diff: {max_diff:.6f}")
    if max_diff < 1e-2:
        print("  CPU and GPU agree — issue is in FlamingoChessLM's layer iteration")
    else:
        print("  CPU and GPU disagree — bfloat16 numerical drift between devices")


def decoder_impl_test(model: FlamingoChessLM) -> None:
    """With all tanh gates == 0, different enc_hidden inputs must produce identical output.
    When gates are closed, tanh(0)=0 so the x-attn contribution is exactly zero — the
    encoder cannot influence the residual stream regardless of enc_hidden content."""
    print("=== decoder_impl_test ===")
    _zero_gates(model)

    B, S = 2, 16
    dev = next(model.parameters()).device
    input_ids    = torch.randint(0, 1000, (B, S), device=dev)
    enc_hidden_a = torch.randn(B, 16, 64, 1024, dtype=torch.bfloat16, device=dev)
    enc_hidden_b = torch.randn(B, 16, 64, 1024, dtype=torch.bfloat16, device=dev)

    with torch.no_grad():
        logits_a = model(input_ids, enc_hidden_a)
        logits_b = model(input_ids, enc_hidden_b)

    max_diff = (logits_a - logits_b).abs().max().item()
    print(f"  Max logit diff between different enc inputs with gates=0: {max_diff}")
    assert max_diff == 0.0, f"Gates=0 but enc_hidden still affects output (max_diff={max_diff})"
    print("  enc_hidden has zero effect when gates are closed  OK")


def gradient_flow_test(model: FlamingoChessLM) -> None:
    """Gradients must reach x_attn params and must NOT reach frozen decoder params."""
    print("=== gradient_flow_test ===")
    _open_gates(model, 0.1)
    model.zero_grad()

    B, S = 2, 16
    dev = next(model.parameters()).device
    input_ids  = torch.randint(0, 1000, (B, S), device=dev)
    enc_hidden = torch.randn(B, 16, 64, 1024, dtype=torch.bfloat16, device=dev)

    logits = model(input_ids, enc_hidden)
    logits.mean().backward()

    # x_attn params must have gradients
    missing = [n for n, p in model.x_attn_layers.named_parameters() if p.grad is None]
    assert not missing, f"x_attn params with no gradient: {missing}"
    print(f"  All {sum(1 for _ in model.x_attn_layers.parameters())} x_attn params have gradients  OK")

    # Frozen decoder params must NOT have gradients
    leaking = [n for n, p in model.decoder.named_parameters() if p.grad is not None]
    assert not leaking, f"Frozen decoder params have gradients (first 5): {leaking[:5]}"
    print("  All decoder params are frozen (no gradients)  OK")

    _zero_gates(model)
    model.zero_grad()


def gate_opens_test(model: FlamingoChessLM) -> None:
    """Different enc_hidden inputs must produce different logits when gates are open."""
    print("=== gate_opens_test ===")
    _open_gates(model, 1.0)

    B, S = 2, 16
    dev = next(model.parameters()).device
    input_ids    = torch.randint(0, 1000, (B, S), device=dev)
    enc_hidden_a = torch.randn(B, 16, 64, 1024, dtype=torch.bfloat16, device=dev)
    enc_hidden_b = torch.randn(B, 16, 64, 1024, dtype=torch.bfloat16, device=dev)

    with torch.no_grad():
        logits_a = model(input_ids, enc_hidden_a)
        logits_b = model(input_ids, enc_hidden_b)

    max_diff = (logits_a - logits_b).abs().max().item()
    print(f"  Max logit diff between different enc inputs: {max_diff:.4f}")
    assert max_diff > 1e-2, f"Logits suspiciously identical with open gates (max_diff={max_diff})"
    print("  Encoder signal reaches decoder  OK")

    _zero_gates(model)


def encoder_impl_test(model: FlamingoChessLM, encoder: Lc0Bt4HFModel) -> None:
    """End-to-end: real positions → LC0 encoder → FlamingoChessLM."""
    print("=== encoder_impl_test ===")

    positions = []
    with open(POSITIONS_PATH) as f:
        for line in f:
            positions.append(json.loads(line))
            if len(positions) == N_POSITIONS:
                break
    print(f"  Loaded {len(positions)} positions from {POSITIONS_PATH}")

    with torch.no_grad():
        enc_hidden = _encode_positions(encoder, positions)
    print(f"  Encoder hidden states shape: {tuple(enc_hidden.shape)}")
    assert enc_hidden.shape == (len(positions), 16, 64, 1024)

    B, S = len(positions), 32
    dev = next(model.parameters()).device
    input_ids = torch.randint(0, 1000, (B, S), device=dev)

    with torch.no_grad():
        logits = model(input_ids, enc_hidden.to(device=dev, dtype=torch.bfloat16))

    assert logits.shape == (B, S, model.decoder.config.vocab_size + model.n_new_tokens)
    print(f"  Logits shape: {tuple(logits.shape)}  OK")


def encoder_benchmark(n: int = 100) -> None:
    """Encode n positions on CPU vs GPU and report throughput."""
    import time
    print(f"=== encoder_benchmark (n={n}) ===")

    positions = []
    with open(POSITIONS_PATH) as f:
        for line in f:
            positions.append(json.loads(line))
            if len(positions) == n:
                break
    print(f"  Loaded {len(positions)} positions")

    def _run(enc, positions, label):
        enc.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            _encode_positions(enc, positions)
            t1 = time.perf_counter()
        elapsed = t1 - t0
        print(f"  {label}: {elapsed:.2f}s  ({len(positions)/elapsed:.1f} pos/s)")

    cpu_encoder = Lc0Bt4HFModel.from_pretrained(ENCODER_PATH, local_files_only=True)
    _run(cpu_encoder, positions, "CPU")

    if torch.cuda.is_available():
        gpu_encoder = Lc0Bt4HFModel.from_pretrained(ENCODER_PATH, local_files_only=True)
        gpu_encoder.to("cuda")
        _run(gpu_encoder, positions, "GPU")
    else:
        print("  GPU not available, skipping GPU benchmark")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def new_token_tests(model: FlamingoChessLM) -> None:
    """Tests for split-embedding new-token functionality."""
    frozen_vocab = model.decoder.config.vocab_size
    dev  = model.device
    B, S = 2, 16

    # ------------------------------------------------------------------
    # Weight tying
    # ------------------------------------------------------------------
    print("=== new_token_weight_tying_test ===")
    assert model.new_embed   is not None, "new_embed not created"
    assert model.new_lm_head is not None, "new_lm_head not created"
    assert model.new_lm_head.weight is model.new_embed.weight, \
        "new_lm_head.weight is NOT tied to new_embed.weight"
    print(f"  new_lm_head.weight is new_embed.weight  OK")

    # ------------------------------------------------------------------
    # Logits shape includes new tokens
    # ------------------------------------------------------------------
    print("=== new_token_logits_shape_test ===")
    input_ids  = torch.randint(0, frozen_vocab, (B, S), device=dev)
    enc_hidden = torch.randn(B, 16, 64, 1024, dtype=torch.bfloat16, device=dev)
    with torch.no_grad():
        logits = model(input_ids, enc_hidden)
    expected = (B, S, frozen_vocab + N_NEW)
    assert logits.shape == expected, f"Expected {expected}, got {tuple(logits.shape)}"
    print(f"  logits shape {tuple(logits.shape)}  OK")

    # ------------------------------------------------------------------
    # Routing: modifying new_embed changes output for new-token inputs
    # ------------------------------------------------------------------
    print("=== new_token_routing_test ===")
    new_ids    = torch.randint(frozen_vocab, frozen_vocab + N_NEW, (B, S), device=dev)
    enc_hidden = torch.randn(B, 16, 64, 1024, dtype=torch.bfloat16, device=dev)
    with torch.no_grad():
        logits_before = model(new_ids, enc_hidden).clone()
        # Zero out all new_embed rows → if routing is correct, output must change
        saved = model.new_embed.weight.data.clone()
        model.new_embed.weight.data.zero_()
        logits_after = model(new_ids, enc_hidden)
        model.new_embed.weight.data.copy_(saved)
    assert not torch.allclose(logits_before, logits_after), \
        "Zeroing new_embed had no effect — new tokens not routed correctly"
    print("  new token IDs routed through new_embed  OK")

    # ------------------------------------------------------------------
    # Gradient flow: new_embed gets grad, frozen decoder does not
    # ------------------------------------------------------------------
    print("=== new_token_gradient_test ===")
    _open_gates(model, 0.1)
    model.zero_grad()

    # Mix of old and new token IDs in input; labels target new tokens
    old_ids = torch.randint(0,            frozen_vocab,           (B, S // 2), device=dev)
    new_ids = torch.randint(frozen_vocab, frozen_vocab + N_NEW,   (B, S // 2), device=dev)
    input_ids = torch.cat([old_ids, new_ids], dim=1)             # (B, S)
    labels    = torch.randint(frozen_vocab, frozen_vocab + N_NEW, (B, S), device=dev)
    enc_hidden = torch.randn(B, 16, 64, 1024, dtype=torch.bfloat16, device=dev)

    logits = model(input_ids, enc_hidden)                         # (B, S, V+77)
    loss   = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
    )
    loss.backward()

    # new_embed must have gradient (covers new_lm_head too since weights are tied)
    assert model.new_embed.weight.grad is not None, "new_embed.weight has no gradient"
    assert model.new_embed.weight.grad.abs().sum() > 0, "new_embed.weight gradient is all zero"
    print(f"  new_embed.weight gradient non-zero  OK")

    # Frozen decoder must not have gradient
    leaking = [n for n, p in model.decoder.named_parameters() if p.grad is not None]
    assert not leaking, f"Frozen decoder params have gradients: {leaking[:3]}"
    print("  Frozen decoder has no gradients  OK")

    # x_attn_layers must have gradient
    missing = [n for n, p in model.x_attn_layers.named_parameters() if p.grad is None]
    assert not missing, f"x_attn params missing gradients: {missing[:3]}"
    print(f"  All x_attn params have gradients  OK")

    _zero_gates(model)
    model.zero_grad()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading FlamingoChessLM (n_new_tokens={N_NEW})...")
    model = FlamingoChessLM.from_pretrained(
        DECODER_PATH, n_new_tokens=N_NEW, device=device, torch_dtype=torch.bfloat16
    )
    model.eval()
    print(f"  model.device = {model.device}")

    print("Loading LC0 encoder...")
    encoder = Lc0Bt4HFModel.from_pretrained(ENCODER_PATH, local_files_only=True)
    encoder.to(device).eval()

    print()
    cpu_gpu_decoder_test()
    print()
    basic_impl_test(model)
    print()
    decoder_impl_test(model)
    print()
    gradient_flow_test(model)
    print()
    gate_opens_test(model)
    print()
    encoder_impl_test(model, encoder)
    print()
    new_token_tests(model)
    print()
    encoder_benchmark()
    print()
    print("All tests passed!")


if __name__ == "__main__":
    main()
