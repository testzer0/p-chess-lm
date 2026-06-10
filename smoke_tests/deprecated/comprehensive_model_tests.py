"""Comprehensive correctness tests for FlamingoChessLM Stage 1.

Run all:                  python chesslm/smoke_tests/comprehensive_model_tests.py
Run one test by name:     python chesslm/smoke_tests/comprehensive_model_tests.py test_zero_gates_matches_raw_decoder_bitexact
Skip model-loading tests: python chesslm/smoke_tests/comprehensive_model_tests.py --no-model

Precision (Group B only):
  --dtype bfloat16   (default — matches training)
  --dtype float32    (use for the bit-exact tests; fp32 has ~7 decimal digits)
  --dtype float64    (slow; for diagnostic deep-checks only)

Tolerances in equivalence tests scale with dtype.

Tests are grouped:
  Group A — no model load required  (cheap, run first; sanity checks)
  Group B — model load required     (load FlamingoChessLM + LC0 once, reuse)
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoTokenizer

from chesslm.encoder.lc0_hf_bt5.hf_model import Lc0Bt4HFModel
from chesslm.models import FlamingoChessLM
from chesslm.train import ntp_loss
from chesslm.utils.eval_utils import (
    _batched_decode,
    _is_consistent,
    _is_valid_parse_tag,
    _sample_next_token,
)
from chesslm.utils.training_utils import collate_fn, init_special_token_embeddings
from chesslm.utils.utils import (
    ANSWER_SPECIAL_TOKENS,
    EMPTY_TOKEN,
    PIECE_TOKENS,
    SQUARE_TOKENS,
    SYSTEM_PROMPT,
    encode_positions,
)

DECODER_PATH = '/scratch/gpfs/DANQIC/jeff/models/smollm-3b-instruct'
ENCODER_PATH = '/scratch/gpfs/DANQIC/jeff/chesslm/chesslm/encoder/lc0_hf_bt5'
N_NEW = len(ANSWER_SPECIAL_TOKENS)

# Dtype config — set by CLI, defaults to bfloat16. Read by _get_model() and
# the tests so tolerances scale appropriately.
DTYPE = torch.bfloat16

_DTYPE_MAP = {
    'bfloat16': torch.bfloat16,
    'bf16':     torch.bfloat16,
    'float16':  torch.float16,
    'fp16':     torch.float16,
    'float32':  torch.float32,
    'fp32':     torch.float32,
    'float64':  torch.float64,
    'fp64':     torch.float64,
}

# Per-dtype relative tolerances for equivalence tests.
# bf16: 7-bit mantissa, ~0.4% relative noise per matmul; cumulative across 36
# layers + cross-attn can reach a few percent. fp32: ~1e-6 per op. fp64: bit-exact.
_REL_TOL = {
    torch.bfloat16: 5e-2,
    torch.float16:  5e-2,
    torch.float32:  1e-4,
    torch.float64:  0.0,
}
_ABS_TOL = {
    torch.bfloat16: 5e-2,
    torch.float16:  5e-2,
    torch.float32:  1e-3,
    torch.float64:  0.0,
}


# ----------------------------------------------------------------------------
# Lazy model loader (shared across model-using tests)
# ----------------------------------------------------------------------------

_cached = {}

def _get_model():
    """Loads / returns (model, tokenizer, encoder, device). Reloads if DTYPE changed."""
    if _cached.get('dtype') == DTYPE:
        return _cached['model'], _cached['tok'], _cached['enc'], _cached['dev']

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  [setup] loading model / tokenizer / encoder on {dev} (dtype={DTYPE})...')

    tok = AutoTokenizer.from_pretrained(DECODER_PATH, local_files_only=True)
    tok.add_tokens(ANSWER_SPECIAL_TOKENS, special_tokens=True)

    model = FlamingoChessLM.from_pretrained(
        DECODER_PATH, n_new_tokens=N_NEW, device=dev, torch_dtype=DTYPE,
        local_files_only=True,
    )
    model.eval()

    enc = Lc0Bt4HFModel.from_pretrained(ENCODER_PATH, local_files_only=True)
    enc.to(device=dev, dtype=DTYPE).eval()

    _cached.update(model=model, tok=tok, enc=enc, dev=dev, dtype=DTYPE)
    return model, tok, enc, dev


def _zero_gates(model, value=0.0):
    for layer in model.x_attn_layers:
        layer.alpha_attn.data.fill_(value)
        layer.alpha_ffn.data.fill_(value)

def _open_gates(model, value=1.0):
    _zero_gates(model, value)


# ============================================================================
# GROUP A — Tests not requiring model load
# ============================================================================

def test_infinite_loader_reshuffles_unlike_cycle():
    """chain.from_iterable(repeat(loader)) must re-run the sampler each epoch; itertools.cycle doesn't."""
    print('=== test_infinite_loader_reshuffles_unlike_cycle ===')
    torch.manual_seed(0)
    ds = TensorDataset(torch.arange(10).float())
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    # itertools.cycle: epoch 2 should replay epoch 1 batch-for-batch
    it_cycle = iter(itertools.cycle(dl))
    cyc = [next(it_cycle)[0].tolist() for _ in range(10)]
    assert cyc[:5] == cyc[5:], f'cycle SHOULD replay; got {cyc}'

    # chain.from_iterable(repeat(...)): epoch 2 should differ from epoch 1 (with shuffle=True)
    it_inf = itertools.chain.from_iterable(itertools.repeat(dl))
    inf = [next(it_inf)[0].tolist() for _ in range(10)]
    assert inf[:5] != inf[5:], f'chain.from_iterable(repeat(...)) should reshuffle; got {inf}'
    print(f'  cycle replayed: {cyc[:5]} == {cyc[5:]}')
    print(f'  inf reshuffled: {inf[:5]} != {inf[5:]}  OK')


def test_smollm3_has_no_sliding_window_layers():
    """model.forward builds ONE causal mask — assert SmolLM3 has only full_attention layers."""
    print('=== test_smollm3_has_no_sliding_window_layers ===')
    cfg = AutoConfig.from_pretrained(DECODER_PATH, local_files_only=True)
    layer_types = getattr(cfg, 'layer_types', None)
    if layer_types is None:
        print('  no layer_types in config (older transformers); skipping')
        return
    bad = [(i, lt) for i, lt in enumerate(layer_types) if lt != 'full_attention']
    assert not bad, (
        f'SmolLM3 has non-full-attention layers: {bad}. '
        'model.forward only builds ONE causal mask and passes it to every layer — '
        'fix model.forward to build per-attention-type masks before training.'
    )
    print(f'  all {len(layer_types)} layers are full_attention  OK')


def test_tokenizer_extends_by_exactly_77():
    print('=== test_tokenizer_extends_by_exactly_77 ===')
    tok = AutoTokenizer.from_pretrained(DECODER_PATH, local_files_only=True)
    orig = len(tok)
    n_added = tok.add_tokens(ANSWER_SPECIAL_TOKENS, special_tokens=True)
    assert n_added == N_NEW == 77, f'expected {N_NEW} new, got {n_added}'
    assert len(tok) - orig == N_NEW, f'len diff = {len(tok)-orig}'
    # New IDs must be contiguous and start at orig
    ids = [tok.convert_tokens_to_ids(t) for t in ANSWER_SPECIAL_TOKENS]
    assert ids == list(range(orig, orig + N_NEW)), f'new IDs not contiguous: first={ids[0]}, last={ids[-1]}, orig={orig}'
    print(f'  added {n_added} tokens, contiguous IDs {orig}..{orig + N_NEW - 1}  OK')


def test_tokenizer_vocab_equals_decoder_vocab_size():
    """The frozen_vocab boundary in model.forward assumes len(tokenizer) == config.vocab_size."""
    print('=== test_tokenizer_vocab_equals_decoder_vocab_size ===')
    tok = AutoTokenizer.from_pretrained(DECODER_PATH, local_files_only=True)
    cfg = AutoConfig.from_pretrained(DECODER_PATH, local_files_only=True)
    assert len(tok) == cfg.vocab_size, (
        f'len(tokenizer)={len(tok)} != config.vocab_size={cfg.vocab_size}. '
        'New-token IDs would either overlap padded rows or skip them.'
    )
    print(f'  len(tokenizer) == config.vocab_size == {cfg.vocab_size}  OK')


def test_chat_template_prompt_is_prefix_of_full():
    """The label boundary `prompt_len` = len(prompt-only template w/ add_generation_prompt)
    requires the prompt-only render to be an exact token-id prefix of the full render."""
    print('=== test_chat_template_prompt_is_prefix_of_full ===')
    tok = AutoTokenizer.from_pretrained(DECODER_PATH, local_files_only=True)
    sys_p = SYSTEM_PROMPT
    q = 'What piece is on e4?'
    a = 'There is a black knight on e4. <PIECE_BN>'
    full = tok.apply_chat_template(
        [{'role': 'system', 'content': sys_p},
         {'role': 'user', 'content': q},
         {'role': 'assistant', 'content': a}],
        tokenize=True, add_generation_prompt=False,
    )
    prompt = tok.apply_chat_template(
        [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': q}],
        tokenize=True, add_generation_prompt=True,
    )
    assert full[:len(prompt)] == prompt, (
        'prompt-only render is NOT a prefix of full render. '
        'Label masking via prompt_len = len(prompt) is WRONG and answer tokens may be mis-aligned.'
    )
    suffix = tok.decode(full[len(prompt):])
    assert 'PIECE_BN' in suffix, f'answer not in suffix: {suffix!r}'
    print(f'  prompt is prefix of full; label suffix = {suffix!r}  OK')


def test_encoder_returns_16_layers():
    print('=== test_encoder_returns_16_layers ===')
    enc = Lc0Bt4HFModel.from_pretrained(ENCODER_PATH, local_files_only=True)
    enc.eval()
    planes = enc.input_planes_from_fen(chess.STARTING_FEN).unsqueeze(0)
    with torch.no_grad():
        out = enc(planes, output_hidden_states=True)
    assert len(out.all_hidden_states) == 16, f'expected 16, got {len(out.all_hidden_states)}'
    for i, h in enumerate(out.all_hidden_states):
        assert h.shape == (1, 64, 1024), f'layer {i}: shape={h.shape}'
    print(f'  16 hidden states, each (B=1, 64, 1024)  OK')


def test_gradscaler_disabled_path_is_noop():
    """train.py enables GradScaler only for fp16. Verify the bf16/fp32 path
    (`enabled=False`) is a safe no-op for the unscale_/step/update calls."""
    print('=== test_gradscaler_disabled_path_is_noop ===')
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.ones(2)
    opt = torch.optim.SGD([p], lr=0.1)
    s = torch.amp.GradScaler(device='cpu', enabled=False)
    loss = (p * 2).sum()
    scaled = s.scale(loss)
    assert (scaled == loss).all().item(), 'scale should be identity when disabled'
    s.unscale_(opt)
    assert (p.grad == 1.0).all().item(), 'unscale_ should not change grads when disabled'
    s.step(opt)
    s.update()
    assert torch.allclose(p, torch.tensor([-0.1, -0.1])), f'optimizer step did not happen: {p}'
    print(f'  disabled GradScaler is a safe no-op  OK')


def test_ntp_loss_ignores_neg_100():
    print('=== test_ntp_loss_ignores_neg_100 ===')
    torch.manual_seed(0)
    logits = torch.randn(2, 10, 100)
    labels = torch.randint(0, 100, (2, 10))
    loss_full = ntp_loss(logits, labels)
    # Mask first half
    masked = labels.clone()
    masked[:, :5] = -100
    loss_partial = ntp_loss(logits, masked)
    assert torch.isfinite(loss_partial), 'partial-mask loss must be finite'
    assert not torch.allclose(loss_full, loss_partial), 'masking should change loss value'
    # All masked → no valid tokens → NaN (PyTorch convention)
    all_masked = torch.full_like(labels, -100)
    loss_none = ntp_loss(logits, all_masked)
    assert torch.isnan(loss_none) or loss_none.item() == 0.0, \
        f'all-masked loss = {loss_none}; expected NaN or 0'
    print(f'  full={loss_full:.4f}, partial={loss_partial:.4f}, none={loss_none}  OK')


def test_ntp_loss_shifts_by_one():
    """A label at position k is matched against logits at position k-1 (next-token)."""
    print('=== test_ntp_loss_shifts_by_one ===')
    V = 50
    logits = torch.full((1, 5, V), -10.0)
    logits[0, 2, 7] = 10.0  # position 2 strongly predicts token 7

    # Label at position 3 means: "logits[2] predicts label[3]" — correct
    labels = torch.full((1, 5), -100)
    labels[0, 3] = 7
    loss_shifted = ntp_loss(logits, labels)
    assert loss_shifted.item() < 1e-2, f'correct shift loss should be tiny, got {loss_shifted}'

    # Label at position 2 means: "logits[1] predicts label[2]" — logits[1] is uniform, high loss
    labels = torch.full((1, 5), -100)
    labels[0, 2] = 7
    loss_unshifted = ntp_loss(logits, labels)
    assert loss_unshifted.item() > 1.0, f'unshifted loss should be large, got {loss_unshifted}'

    print(f'  shifted (correct): {loss_shifted:.4f}; unshifted: {loss_unshifted:.4f}  OK')


def test_top_p_sampling_keeps_at_least_one():
    print('=== test_top_p_sampling_keeps_at_least_one ===')
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    # Extreme top_p — without the safety guard would mask everything
    out = _sample_next_token(logits, temperature=1.0, top_k=0, top_p=1e-6)
    assert out.shape == (4,)
    assert (out >= 0).all() and (out < 50).all(), f'out-of-range: {out}'
    print(f'  top_p=1e-6 still produces valid IDs: {out.tolist()}  OK')


def test_top_k_one_always_picks_argmax():
    print('=== test_top_k_one_always_picks_argmax ===')
    logits = torch.zeros(1, 10)
    logits[0, 5] = 100.0
    g_greedy = _sample_next_token(logits, 0.0, 0, 1.0).item()
    assert g_greedy == 5
    torch.manual_seed(0)
    picks = [_sample_next_token(logits, 1.0, 1, 1.0).item() for _ in range(20)]
    assert all(p == 5 for p in picks), f'top_k=1 deviated from argmax: {picks}'
    print(f'  greedy & top_k=1 both consistently pick argmax  OK')


def test_parse_validation_static_square():
    print('=== test_parse_validation_static_square ===')
    qt = 'static_square'
    assert _is_valid_parse_tag(['<SQUARE_E4>', '<PIECE_BN>'], qt)
    assert _is_valid_parse_tag(['<SQUARE_A1>', '<EMPTY>'], qt)
    assert not _is_valid_parse_tag(['<PIECE_BN>', '<SQUARE_E4>'], qt), 'wrong order'
    assert not _is_valid_parse_tag(['<SQUARE_E4>'], qt), 'too short'
    assert not _is_valid_parse_tag([], qt), 'empty'
    assert not _is_valid_parse_tag(['<SQUARE_E4>', '<SQUARE_F4>'], qt), 'two squares'
    print('  static_square parse validation correct  OK')


def test_parse_validation_static_piece():
    print('=== test_parse_validation_static_piece ===')
    qt = 'static_piece'
    assert _is_valid_parse_tag(['<PIECE_WP>', '<SQUARE_A2>', '<SQUARE_B2>'], qt)
    assert _is_valid_parse_tag(['<PIECE_WP>', '<EMPTY>'], qt)
    assert not _is_valid_parse_tag(['<PIECE_WP>'], qt), 'no squares listed'
    assert not _is_valid_parse_tag(['<SQUARE_A1>', '<SQUARE_B1>'], qt), 'first must be piece'
    assert not _is_valid_parse_tag(['<PIECE_WP>', '<PIECE_BN>'], qt), 'rest must be squares or EMPTY'
    print('  static_piece parse validation correct  OK')


def test_consistency_static_square():
    print('=== test_consistency_static_square ===')
    fen = chess.STARTING_FEN
    assert _is_consistent('static_square', ['<SQUARE_E4>', '<EMPTY>'], fen)
    assert _is_consistent('static_square', ['<SQUARE_A2>', '<PIECE_WP>'], fen)
    assert _is_consistent('static_square', ['<SQUARE_E1>', '<PIECE_WK>'], fen)
    assert not _is_consistent('static_square', ['<SQUARE_A2>', '<PIECE_BP>'], fen), 'wrong color'
    assert not _is_consistent('static_square', ['<SQUARE_E4>', '<PIECE_WP>'], fen), 'square is empty'
    print('  static_square consistency correct  OK')


def test_consistency_static_piece():
    print('=== test_consistency_static_piece ===')
    fen = chess.STARTING_FEN
    expected_wp = [f'<SQUARE_{chr(ord("A") + f)}2>' for f in range(8)]
    assert _is_consistent('static_piece', ['<PIECE_WP>'] + expected_wp, fen)
    # Out-of-order squares should still match (sorted comparison)
    assert _is_consistent('static_piece', ['<PIECE_WP>'] + list(reversed(expected_wp)), fen)
    assert not _is_consistent('static_piece', ['<PIECE_WP>'] + expected_wp[:-1], fen), 'missing one'

    empty_fen = '8/8/8/8/8/8/8/k6K w - - 0 1'
    assert _is_consistent('static_piece', ['<PIECE_WQ>', '<EMPTY>'], empty_fen)
    assert not _is_consistent('static_piece', ['<PIECE_WQ>', '<SQUARE_D1>'], empty_fen)
    assert not _is_consistent('static_piece', ['<PIECE_WK>', '<EMPTY>'], empty_fen), 'WK is on h1'
    assert _is_consistent('static_piece', ['<PIECE_WK>', '<SQUARE_H1>'], empty_fen)
    print('  static_piece consistency correct  OK')


def test_x_attn_positions_match_spec():
    print('=== test_x_attn_positions_match_spec ===')
    assert FlamingoChessLM.X_ATTN_POSITIONS == list(range(0, 32, 2)), \
        f'X_ATTN_POSITIONS = {FlamingoChessLM.X_ATTN_POSITIONS}'
    assert FlamingoChessLM.N_XATTN == 16
    assert FlamingoChessLM.ENCODER_DIM == 1024
    assert FlamingoChessLM.DECODER_DIM == 2048
    print('  X_ATTN_POSITIONS=[0,2,...,30]; N_XATTN=16; ENC=1024, DEC=2048  OK')


def test_answer_special_tokens_layout():
    print('=== test_answer_special_tokens_layout ===')
    assert len(SQUARE_TOKENS) == 64
    assert len(PIECE_TOKENS) == 12
    assert ANSWER_SPECIAL_TOKENS == SQUARE_TOKENS + PIECE_TOKENS + [EMPTY_TOKEN]
    assert len(ANSWER_SPECIAL_TOKENS) == 77 == N_NEW
    # Square names: a1, b1, ..., h8 in chess.SQUARES order (a1=0)
    assert SQUARE_TOKENS[0] == '<SQUARE_A1>'
    assert SQUARE_TOKENS[63] == '<SQUARE_H8>'
    print('  64 squares + 12 pieces + 1 empty = 77 special tokens  OK')


# ============================================================================
# GROUP B — Tests requiring model load
# ============================================================================

def test_encode_positions_canonicalization_white():
    """White-to-move positions are NOT reindexed by encode_positions."""
    print('=== test_encode_positions_canonicalization_white ===')
    _, _, enc, dev = _get_model()
    fen = chess.STARTING_FEN

    planes = enc.input_planes_from_fen(fen).unsqueeze(0).to(dev)
    with torch.no_grad():
        raw = torch.stack(enc(planes, output_hidden_states=True).all_hidden_states, dim=1)

    canon = encode_positions(enc, [fen], [[]], [fen], dev, torch.float32)
    assert torch.allclose(canon, raw.float()), 'white-to-move should not be reindexed'
    print('  white-to-move: canon == raw  OK')


def test_encode_positions_canonicalization_black():
    """Black-to-move: canon[..., sq] == raw[..., sq ^ 56]."""
    print('=== test_encode_positions_canonicalization_black ===')
    _, _, enc, dev = _get_model()

    start = chess.STARTING_FEN
    b = chess.Board()
    b.push(chess.Move.from_uci('e2e4'))
    end_fen = b.fen()
    assert chess.Board(end_fen).turn == chess.BLACK

    planes = enc.input_planes_from_fen(start, ['e2e4']).unsqueeze(0).to(dev)
    with torch.no_grad():
        raw = torch.stack(enc(planes, output_hidden_states=True).all_hidden_states, dim=1).float()

    canon = encode_positions(enc, [start], [['e2e4']], [end_fen], dev, torch.float32)
    for sq in range(64):
        diff = (canon[0, :, sq] - raw[0, :, sq ^ 56]).abs().max().item()
        assert diff == 0.0, f'sq={sq}: canon vs raw[sq^56] diff={diff}'
    print('  black-to-move: canon[sq] == raw[sq^56] for all 64 squares  OK')


def test_forward_shape_and_routing():
    """Output shape includes new tokens; routing of new-token IDs works."""
    print('=== test_forward_shape_and_routing ===')
    model, _, _, dev = _get_model()
    _zero_gates(model)

    B, S = 2, 12
    frozen_vocab = model.decoder.config.vocab_size
    # Mix of old and new IDs
    input_ids = torch.randint(0, frozen_vocab + N_NEW, (B, S), device=dev)
    enc_hidden = torch.zeros(B, 16, 64, 1024, dtype=DTYPE, device=dev)
    with torch.no_grad():
        logits = model(input_ids, enc_hidden)
    expected = (B, S, frozen_vocab + N_NEW)
    assert logits.shape == expected, f'got {tuple(logits.shape)}'

    # Zeroing new_embed must change output if any new-token IDs were used
    new_only = torch.randint(frozen_vocab, frozen_vocab + N_NEW, (B, S), device=dev)
    with torch.no_grad():
        before = model(new_only, enc_hidden).clone()
        saved = model.new_embed.weight.data.clone()
        model.new_embed.weight.data.zero_()
        after = model(new_only, enc_hidden)
        model.new_embed.weight.data.copy_(saved)
    assert not torch.allclose(before, after), 'new-token routing broken'
    print('  shape OK; new-token routing through new_embed OK')


def test_zero_gates_matches_raw_decoder_bitexact():
    """With all gates = 0, FlamingoChessLM output (restricted to frozen vocab) must match
    raw SmolLM3 output. fp64 → bit-exact. fp32 → tight tolerance. bf16 → looser
    (accumulation order in the extra x-attn computation introduces noise even when
    its contribution is gated to zero arithmetically)."""
    print(f'=== test_zero_gates_matches_raw_decoder_bitexact (dtype={DTYPE}) ===')
    model, _, _, dev = _get_model()
    _zero_gates(model)
    B, S = 2, 16
    frozen_vocab = model.decoder.config.vocab_size
    input_ids = torch.randint(0, frozen_vocab, (B, S), device=dev)
    enc_hidden = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)

    with torch.no_grad():
        flam = model(input_ids, enc_hidden)[..., :frozen_vocab]
        raw = model.decoder(input_ids).logits

    abs_diff = (flam - raw).abs().max().item()
    scale = max(raw.abs().max().item(), 1.0)
    rel_diff = abs_diff / scale
    abs_tol = _ABS_TOL[DTYPE]
    rel_tol = _REL_TOL[DTYPE]
    assert abs_diff <= abs_tol or rel_diff <= rel_tol, (
        f'gates=0 path drifts from raw decoder: abs={abs_diff:.6f}, '
        f'rel={rel_diff:.6f}, tol abs={abs_tol}, rel={rel_tol}'
    )
    # Argmax over the frozen-vocab logits must match regardless of dtype
    am_diff = (flam.argmax(-1) != raw.argmax(-1)).float().mean().item()
    assert am_diff < 0.02, f'argmax disagreement rate {am_diff:.3f} too high'
    print(f'  abs_diff={abs_diff:.6f}, rel_diff={rel_diff:.6f}, argmax_disagree={am_diff:.4f}  OK')


def test_gates_closed_blocks_encoder_signal():
    """Different encoder hidden states produce identical output when all gates = 0."""
    print('=== test_gates_closed_blocks_encoder_signal ===')
    model, _, _, dev = _get_model()
    _zero_gates(model)
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S), device=dev)
    enc_a = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)
    enc_b = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)
    with torch.no_grad():
        la = model(input_ids, enc_a)
        lb = model(input_ids, enc_b)
    diff = (la - lb).abs().max().item()
    assert diff == 0.0, f'closed gates leak encoder signal; diff={diff}'
    print('  closed gates fully block encoder signal  OK')


def test_gates_open_propagates_encoder_signal():
    print('=== test_gates_open_propagates_encoder_signal ===')
    model, _, _, dev = _get_model()
    _open_gates(model, 1.0)
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S), device=dev)
    enc_a = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)
    enc_b = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)
    with torch.no_grad():
        la = model(input_ids, enc_a)
        lb = model(input_ids, enc_b)
    diff = (la - lb).abs().max().item()
    assert diff > 1e-2, f'open gates fail to propagate; diff={diff}'
    _zero_gates(model)
    print(f'  open gates propagate; logit diff = {diff:.4f}  OK')


def test_left_pad_position_ids_invariance():
    """Last-position logits must match between unpadded and left-padded inputs when
    position_ids and attn_mask are set correctly. This is the invariant that makes
    batched generation correct."""
    print('=== test_left_pad_position_ids_invariance ===')
    model, _, _, dev = _get_model()
    _open_gates(model, 0.5)

    L = 12
    frozen_vocab = model.decoder.config.vocab_size
    real_ids = torch.randint(0, frozen_vocab, (1, L), device=dev)
    enc_hidden = torch.randn(1, 16, 64, 1024, dtype=DTYPE, device=dev)

    # Unpadded
    attn_a = torch.ones(1, L, dtype=torch.long, device=dev)
    pos_a = torch.arange(L, device=dev).unsqueeze(0)

    # Left-padded (pad_id arbitrary because attn_mask masks it)
    P = 7
    pad_id = 0
    padded_ids = torch.full((1, L + P), pad_id, dtype=torch.long, device=dev)
    padded_ids[0, P:] = real_ids[0]
    attn_b = torch.zeros(1, L + P, dtype=torch.long, device=dev)
    attn_b[0, P:] = 1
    pos_b = (attn_b.cumsum(-1) - 1).clamp(min=0)

    with torch.no_grad():
        la = model(real_ids, enc_hidden, attn_a, position_ids=pos_a)
        lb = model(padded_ids, enc_hidden, attn_b, position_ids=pos_b)

    last_a = la[0, -1].float()
    last_b = lb[0, -1].float()
    abs_diff = (last_a - last_b).abs().max().item()
    rel = abs_diff / max(last_a.abs().max().item(), 1.0)
    assert rel < 0.05, f'last-position logits diverge: abs_diff={abs_diff}, rel={rel}'
    am_a, am_b = last_a.argmax().item(), last_b.argmax().item()
    assert am_a == am_b, f'argmax mismatch: unpadded={am_a}, padded={am_b}'

    _zero_gates(model)
    print(f'  last-pos diff abs={abs_diff:.4f} rel={rel:.4f}, argmax match  OK')


def test_generation_batched_matches_individual():
    """First generated token under left-padded batched generation matches single-example
    generation for both prompts in the batch."""
    print('=== test_generation_batched_matches_individual ===')
    model, tok, _, dev = _get_model()
    _open_gates(model, 0.5)
    torch.manual_seed(0)

    enc_short = torch.randn(1, 16, 64, 1024, dtype=DTYPE, device=dev)
    enc_long = torch.randn(1, 16, 64, 1024, dtype=DTYPE, device=dev)
    enc_batch = torch.cat([enc_short, enc_long], dim=0)

    frozen_vocab = model.decoder.config.vocab_size
    short_ids = torch.randint(0, frozen_vocab, (10,)).tolist()
    long_ids = torch.randint(0, frozen_vocab, (22,)).tolist()

    eos = tok.eos_token_id
    pad = tok.pad_token_id or tok.eos_token_id

    gen_batched = _batched_decode(
        model, enc_batch, [short_ids, long_ids],
        pad, eos, max_new_tokens=3, device=dev, amp_dtype=DTYPE,
        temperature=0.0,
    )
    gen_short_alone = _batched_decode(
        model, enc_short, [short_ids],
        pad, eos, max_new_tokens=3, device=dev, amp_dtype=DTYPE,
        temperature=0.0,
    )
    gen_long_alone = _batched_decode(
        model, enc_long, [long_ids],
        pad, eos, max_new_tokens=3, device=dev, amp_dtype=DTYPE,
        temperature=0.0,
    )

    # First generated token must match (later tokens may bifurcate due to bf16 noise + argmax)
    assert gen_batched[0][0] == gen_short_alone[0][0], (
        f'short first-token mismatch: batched={gen_batched[0][0]} alone={gen_short_alone[0][0]}'
    )
    assert gen_batched[1][0] == gen_long_alone[0][0], (
        f'long first-token mismatch: batched={gen_batched[1][0]} alone={gen_long_alone[0][0]}'
    )
    _zero_gates(model)
    print(f'  batched: {gen_batched}  alone: short={gen_short_alone}, long={gen_long_alone}  OK')


def test_new_embed_weight_tying_persists_after_step():
    print('=== test_new_embed_weight_tying_persists_after_step ===')
    model, _, _, dev = _get_model()
    _open_gates(model, 0.1)
    assert model.new_lm_head.weight is model.new_embed.weight, 'tying broken at init'

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    B, S = 1, 8
    frozen_vocab = model.decoder.config.vocab_size
    input_ids = torch.randint(0, frozen_vocab + N_NEW, (B, S), device=dev)
    labels = torch.randint(frozen_vocab, frozen_vocab + N_NEW, (B, S), device=dev)
    enc_hidden = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)

    logits = model(input_ids, enc_hidden)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad()

    assert model.new_lm_head.weight is model.new_embed.weight, 'tying broken AFTER optimizer step'
    # Also verify both modules see the same .data buffer (post-step verification)
    assert model.new_lm_head.weight.data_ptr() == model.new_embed.weight.data_ptr(), \
        'weights diverged into separate buffers'
    _zero_gates(model)
    print(f'  tying preserved: same Parameter and same data buffer  OK')


def test_frozen_decoder_unchanged_after_step():
    """After backward+step, sampled frozen decoder params are bit-exactly unchanged."""
    print('=== test_frozen_decoder_unchanged_after_step ===')
    model, _, _, dev = _get_model()
    _open_gates(model, 0.1)

    snaps = {n: p.detach().clone() for n, p in list(model.decoder.named_parameters())[:8]}

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)  # large LR to amplify any leak
    B, S = 1, 8
    frozen_vocab = model.decoder.config.vocab_size
    input_ids = torch.randint(0, frozen_vocab + N_NEW, (B, S), device=dev)
    labels = torch.randint(0, frozen_vocab + N_NEW, (B, S), device=dev)
    enc_hidden = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)

    logits = model(input_ids, enc_hidden)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
    loss.backward()

    # Frozen decoder grads should all be None
    leaking = [n for n, p in model.decoder.named_parameters() if p.grad is not None]
    assert not leaking, f'decoder gradients leaked: {leaking[:5]}'

    opt.step()
    opt.zero_grad()

    decoder_params = dict(model.decoder.named_parameters())
    for n, p_old in snaps.items():
        p_new = decoder_params[n]
        assert (p_old == p_new).all().item(), f'frozen param {n} changed after step!'
    _zero_gates(model)
    print(f'  {len(snaps)} sampled frozen params bit-exactly unchanged  OK')


def test_xattn_params_receive_gradients():
    print('=== test_xattn_params_receive_gradients ===')
    model, _, _, dev = _get_model()
    _open_gates(model, 0.1)
    model.zero_grad()

    B, S = 1, 8
    input_ids = torch.randint(0, 1000, (B, S), device=dev)
    enc_hidden = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)
    logits = model(input_ids, enc_hidden)
    logits.mean().backward()

    missing = [n for n, p in model.x_attn_layers.named_parameters() if p.grad is None]
    assert not missing, f'x_attn params without grad: {missing}'
    # And non-zero
    zeros = [n for n, p in model.x_attn_layers.named_parameters() if p.grad.abs().sum() == 0]
    assert not zeros, f'x_attn params with zero grad: {zeros}'
    _zero_gates(model)
    print(f'  all {sum(1 for _ in model.x_attn_layers.parameters())} x_attn params have non-zero grad  OK')


def test_collate_label_masking_and_pad():
    print('=== test_collate_label_masking_and_pad ===')
    _, tok, _, _ = _get_model()
    ex_short = {
        'question': 'What piece is on e4?',
        'answer': 'There is a black knight on e4. <PIECE_BN>',
        'start_fen': chess.STARTING_FEN, 'moves': [], 'fen': chess.STARTING_FEN,
        'question_type': 'static_square', 'answer_class': ['<SQUARE_E4>', '<PIECE_BN>'],
    }
    ex_long = {
        'question': 'List all squares with a white pawn please be verbose.',
        'answer': 'White pawns are on: <PIECE_WP> <SQUARE_A2> <SQUARE_B2>',
        'start_fen': chess.STARTING_FEN, 'moves': [], 'fen': chess.STARTING_FEN,
        'question_type': 'static_piece', 'answer_class': ['<PIECE_WP>', '<SQUARE_A2>', '<SQUARE_B2>'],
    }
    batch = collate_fn([ex_short, ex_long], tokenizer=tok,
                       system_prompt=SYSTEM_PROMPT, max_seq_len=512)
    input_ids = batch['input_ids']
    labels = batch['labels']
    attn = batch['attention_mask']
    assert input_ids.shape == labels.shape == attn.shape, 'shape mismatch'

    for i in range(2):
        L = attn[i].sum().item()
        # Non-pad: labels are either -100 (prompt) or = input_ids[i,j] (answer)
        for j in range(L):
            if labels[i, j].item() != -100:
                assert labels[i, j].item() == input_ids[i, j].item(), \
                    f'label != input at [{i},{j}]'
        # Pad positions
        pad_pos = attn[i] == 0
        assert (labels[i][pad_pos] == -100).all().item(), f'row {i} pads not -100'

    # Unmasked decode contains answer special token text
    unmasked = [t.item() for t in labels[0] if t.item() != -100]
    decoded = tok.decode(unmasked)
    assert '<PIECE_BN>' in decoded, f'PIECE_BN not in unmasked: {decoded!r}'
    print(f'  labels masked correctly; sample decode = {decoded!r}  OK')


def test_semantic_embed_init_changes_new_embed_and_preserves_tying():
    print('=== test_semantic_embed_init_changes_new_embed_and_preserves_tying ===')
    model, tok, _, _ = _get_model()
    before = model.new_embed.weight.data.clone()

    init_special_token_embeddings(model, tok, 'semantic')
    after = model.new_embed.weight.data

    assert not torch.allclose(before, after), 'semantic init did not change new_embed'
    assert model.new_lm_head.weight is model.new_embed.weight, 'tying broken by init'

    # Norms should be in a reasonable range vs frozen embeddings
    new_avg_norm = after.float().norm(dim=-1).mean().item()
    frozen_avg_norm = model.decoder.model.embed_tokens.weight.data.float().norm(dim=-1).mean().item()
    ratio = new_avg_norm / max(frozen_avg_norm, 1e-6)
    assert 0.05 < ratio < 20.0, f'semantic init norm ratio {ratio:.3f} looks suspicious'
    print(f'  new_embed initialized; norm ratio (new/frozen) = {ratio:.3f}  OK')


def test_random_embed_init_is_noop():
    print('=== test_random_embed_init_is_noop ===')
    model, tok, _, _ = _get_model()
    before = model.new_embed.weight.data.clone()
    init_special_token_embeddings(model, tok, 'random')
    assert torch.equal(before, model.new_embed.weight.data), 'random init should be no-op'
    print('  random strategy leaves new_embed untouched  OK')


def test_param_count_roughly_matches_spec():
    """kv_bridge.md claims ~470M trainable x_attn params + new_embed."""
    print('=== test_param_count_roughly_matches_spec ===')
    model, _, _, _ = _get_model()
    xattn = sum(p.numel() for p in model.x_attn_layers.parameters())
    new_embed = sum(p.numel() for p in model.new_embed.parameters())
    total_train = sum(p.numel() for p in model.trainable_parameters())
    print(f'  x_attn: {xattn/1e6:.1f}M, new_embed: {new_embed/1e6:.3f}M, total trainable: {total_train/1e6:.1f}M')
    # Spec ~29.4M * 16 = ~470M for x_attn
    assert 400e6 < xattn < 550e6, f'x_attn param count {xattn/1e6:.1f}M outside expected 400-550M'
    # new_embed: 77 * 2048 = ~158K
    assert new_embed == 77 * 2048, f'new_embed = {new_embed}'
    print('  x_attn ≈ 470M, new_embed = 77*2048  OK')


# ============================================================================
# Runner
# ============================================================================

GROUP_A = [
    test_infinite_loader_reshuffles_unlike_cycle,
    test_smollm3_has_no_sliding_window_layers,
    test_tokenizer_extends_by_exactly_77,
    test_tokenizer_vocab_equals_decoder_vocab_size,
    test_chat_template_prompt_is_prefix_of_full,
    test_encoder_returns_16_layers,
    test_gradscaler_disabled_path_is_noop,
    test_ntp_loss_ignores_neg_100,
    test_ntp_loss_shifts_by_one,
    test_top_p_sampling_keeps_at_least_one,
    test_top_k_one_always_picks_argmax,
    test_parse_validation_static_square,
    test_parse_validation_static_piece,
    test_consistency_static_square,
    test_consistency_static_piece,
    test_x_attn_positions_match_spec,
    test_answer_special_tokens_layout,
]

GROUP_B = [
    test_encode_positions_canonicalization_white,
    test_encode_positions_canonicalization_black,
    test_forward_shape_and_routing,
    test_zero_gates_matches_raw_decoder_bitexact,
    test_gates_closed_blocks_encoder_signal,
    test_gates_open_propagates_encoder_signal,
    test_left_pad_position_ids_invariance,
    test_generation_batched_matches_individual,
    test_new_embed_weight_tying_persists_after_step,
    test_frozen_decoder_unchanged_after_step,
    test_xattn_params_receive_gradients,
    test_collate_label_masking_and_pad,
    test_semantic_embed_init_changes_new_embed_and_preserves_tying,
    test_random_embed_init_is_noop,
    test_param_count_roughly_matches_spec,
]


def main():
    global DTYPE
    args = sys.argv[1:]
    skip_model = '--no-model' in args
    args = [a for a in args if a != '--no-model']

    # --dtype <name>
    if '--dtype' in args:
        i = args.index('--dtype')
        name = args[i + 1]
        if name not in _DTYPE_MAP:
            print(f'Unknown dtype {name!r}; valid: {sorted(set(_DTYPE_MAP))}')
            sys.exit(2)
        DTYPE = _DTYPE_MAP[name]
        del args[i:i + 2]

    target = args[0] if args else None
    print(f'[config] dtype={DTYPE}, skip_model={skip_model}, target={target}')

    tests = list(GROUP_A) + ([] if skip_model else list(GROUP_B))
    if target:
        tests = [t for t in tests if t.__name__ == target]
        if not tests:
            print(f'No test named {target!r}')
            sys.exit(2)

    passed, failed = [], []
    for fn in tests:
        try:
            fn()
            passed.append(fn.__name__)
        except Exception as e:
            import traceback
            print(f'  FAIL: {type(e).__name__}: {e}')
            traceback.print_exc()
            failed.append(fn.__name__)
        print()

    print('=' * 60)
    print(f'PASSED ({len(passed)}):')
    for n in passed:
        print(f'  + {n}')
    if failed:
        print(f'\nFAILED ({len(failed)}):')
        for n in failed:
            print(f'  - {n}')
    print('=' * 60)
    sys.exit(0 if not failed else 1)


if __name__ == '__main__':
    main()
