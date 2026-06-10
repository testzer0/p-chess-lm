"""Comprehensive correctness tests for the chess-LM architectures.

Covers FlamingoChessLM, LLaVAChessLM at lora_rank in {-1, 0, 8}.

Run all:                  python chesslm/smoke_tests/comprehensive_arch_tests.py
Run one test by name:     python chesslm/smoke_tests/comprehensive_arch_tests.py test_apply_lora_rank_negative_freezes
Skip model-loading tests: python chesslm/smoke_tests/comprehensive_arch_tests.py --no-model
Skip Group A only:        python chesslm/smoke_tests/comprehensive_arch_tests.py --no-helpers

Precision:
  --dtype bfloat16   (default — matches training)
  --dtype float32    (use for tight tolerance checks)

Test groups:
  Group A  — shared helpers (no decoder load required, fast)
  Group B  — per-arch tests, each parametrized over lora_rank in {-1, 0, 8}
             For each (arch, lora_rank): init, forward, backward, grad-flow,
             param_groups, save/load round-trip
  Group C  — cross-arch consistency tests (run once, hits real decoder)

Design notes:
- Group B reloads the decoder for each (arch, lora_rank) pair because LoRA
  wrapping mutates the decoder in place. This is the only correct way to
  guarantee parameter isolation between tests.
- We do NOT verify the encoder pipeline here — that is covered separately in
  the deprecated comprehensive_model_tests.py and is orthogonal to lora_rank.
"""

import gc
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from chesslm.models import FlamingoChessLM, LLaVAChessLM
from chesslm.models.base import (
    apply_lora,
    decoder_trainable_params,
    init_new_token_embeddings,
    load_decoder_state,
    save_decoder_state,
    unwrap_decoder,
)
from chesslm.utils.utils import ANSWER_SPECIAL_TOKENS

DECODER_PATH = '/scratch/gpfs/DANQIC/jeff/models/smollm-3b-instruct'
N_NEW = len(ANSWER_SPECIAL_TOKENS)

DTYPE = torch.bfloat16

_DTYPE_MAP = {
    'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
    'float16':  torch.float16,  'fp16': torch.float16,
    'float32':  torch.float32,  'fp32': torch.float32,
}

# Rank values exercised across Group B. -1=frozen, 0=full, 8=LoRA.
RANKS = (-1, 0, 8)


# ============================================================================
# Helpers
# ============================================================================

def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_fresh_decoder(dev):
    """Load a fresh SmolLM3 decoder. Caller is responsible for cleanup."""
    return AutoModelForCausalLM.from_pretrained(
        DECODER_PATH,
        torch_dtype=DTYPE,
        local_files_only=True,
    ).to(dev)


def _build_arch(arch, lora_rank, dev):
    """Build (model, decoder) for the given (arch, lora_rank). Loads a fresh
    decoder each call to guarantee independence — LoRA wraps in place."""
    decoder = _load_fresh_decoder(dev)
    if arch == 'flamingo':
        model = FlamingoChessLM(decoder, n_new_tokens=N_NEW, lora_rank=lora_rank)
    elif arch == 'llava':
        model = LLaVAChessLM(decoder, n_new_tokens=N_NEW, lora_rank=lora_rank)
    else:
        raise ValueError(f'unknown arch {arch!r}')

    # Mimic from_pretrained device/dtype placement of the non-decoder modules
    if arch == 'flamingo':
        model.x_attn_layers.to(device=dev, dtype=DTYPE)
    elif arch == 'llava':
        model.connector.to(device=dev, dtype=DTYPE)
        model.file_embed.to(device=dev, dtype=DTYPE)
        model.rank_embed.to(device=dev, dtype=DTYPE)
    # new_embed / new_lm_head are already allocated on dev/DTYPE inside __init__
    # via init_new_token_embeddings — no post-hoc cast needed (which would sever
    # weight tying). Matches the real from_pretrained flow.
    return model


def _free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_inputs(B, S, dev, frozen_vocab):
    """Build (input_ids, encoder_hidden_states, attention_mask) for forward."""
    input_ids = torch.randint(0, frozen_vocab + N_NEW, (B, S), device=dev)
    enc = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)
    attn = torch.ones(B, S, dtype=torch.long, device=dev)
    return input_ids, enc, attn


def _arch_cls(arch):
    return {'flamingo': FlamingoChessLM, 'llava': LLaVAChessLM}[arch]


# ============================================================================
# GROUP A — shared helper tests (no decoder load)
# ============================================================================

def test_apply_lora_rank_negative_freezes():
    """lora_rank<0 freezes all decoder params in-place and returns decoder unchanged."""
    print('=== test_apply_lora_rank_negative_freezes ===')
    dec = torch.nn.Linear(8, 8)
    for p in dec.parameters():
        p.requires_grad_(True)
    out = apply_lora(dec, -1)
    assert out is dec, 'rank<0 should return decoder unchanged (no wrap)'
    assert all(not p.requires_grad for p in dec.parameters()), 'rank<0 should freeze all params'
    print('  rank=-1: decoder returned as-is, all params requires_grad=False  OK')


def test_apply_lora_rank_zero_passthrough():
    """lora_rank=0 returns decoder unchanged with requires_grad untouched."""
    print('=== test_apply_lora_rank_zero_passthrough ===')
    dec = torch.nn.Linear(8, 8)
    for p in dec.parameters():
        p.requires_grad_(True)
    out = apply_lora(dec, 0)
    assert out is dec
    assert all(p.requires_grad for p in dec.parameters()), \
        'rank=0 should NOT freeze params'
    print('  rank=0: decoder returned as-is, requires_grad preserved  OK')


def test_apply_lora_rank_positive_wraps_with_peft():
    """lora_rank>0 wraps decoder with PEFT and freezes the backbone."""
    print('=== test_apply_lora_rank_positive_wraps_with_peft ===')
    # Build a minimal HF-like model PEFT can wrap (q_proj target).
    class TinyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(16, 16, bias=False)
            self.k_proj = torch.nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.k_proj(self.q_proj(x))

    dec = TinyLayer()
    wrapped = apply_lora(dec, 8, target_modules=['q_proj'])
    assert hasattr(wrapped, 'get_base_model'), 'rank>0 must return a PeftModel'
    # The base q_proj weight should now be frozen
    base = wrapped.get_base_model()
    assert not base.q_proj.weight.requires_grad, 'PEFT should freeze base q_proj.weight'
    # At least one new parameter (LoRA A/B) should be trainable
    trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
    assert any('lora' in n.lower() for n in trainable), \
        f'no lora params trainable: {trainable[:5]}'
    print(f'  rank=8: wrapped, base frozen, {len(trainable)} lora params trainable  OK')


def test_unwrap_decoder_idempotent_for_plain_decoder():
    print('=== test_unwrap_decoder_idempotent_for_plain_decoder ===')
    dec = torch.nn.Linear(4, 4)
    assert unwrap_decoder(dec) is dec
    print('  plain decoder returns self  OK')


def test_unwrap_decoder_peels_peft_wrapper():
    print('=== test_unwrap_decoder_peels_peft_wrapper ===')
    class TinyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.q_proj(x)

    dec = TinyLayer()
    wrapped = apply_lora(dec, 8, target_modules=['q_proj'])
    base = unwrap_decoder(wrapped)
    assert base is dec, 'unwrap should peel back to the original module'
    print('  unwrap(PEFT-wrapped) === original  OK')


def test_decoder_trainable_params_partition():
    """Returns []/all/lora-only depending on lora_rank."""
    print('=== test_decoder_trainable_params_partition ===')
    # rank<0
    dec1 = torch.nn.Linear(4, 4)
    for p in dec1.parameters():
        p.requires_grad_(False)
    assert decoder_trainable_params(dec1, -1) == []

    # rank=0
    dec2 = torch.nn.Linear(4, 4)
    full = decoder_trainable_params(dec2, 0)
    assert len(full) == len(list(dec2.parameters()))

    # rank>0 — only requires_grad=True params
    class L(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.q_proj(x)
    dec3 = L()
    wrapped = apply_lora(dec3, 4, target_modules=['q_proj'])
    params = decoder_trainable_params(wrapped, 4)
    assert all(p.requires_grad for p in params)
    assert len(params) > 0, 'should have at least the LoRA A/B params'
    print(f'  rank=-1 → 0 params; rank=0 → all; rank=4 → {len(params)} (LoRA only)  OK')


def test_save_load_decoder_state_roundtrip_rank_zero():
    """Full state_dict round-trip for rank=0."""
    print('=== test_save_load_decoder_state_roundtrip_rank_zero ===')
    dec = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    # state_dict() returns *references* to live params, so we clone before
    # corrupting — otherwise the saved dict is mutated in place and there's
    # nothing to restore from.
    sd = {k: v.detach().clone() for k, v in save_decoder_state(dec, 0).items()}
    # Corrupt then reload
    for p in dec.parameters():
        p.data.zero_()
    load_decoder_state(dec, 0, sd)
    # Should not be all-zero anymore
    nonzero = any(p.abs().sum().item() > 0 for p in dec.parameters())
    assert nonzero, 'load did not restore'
    print('  full save/load round-trip restores params  OK')


def test_save_load_decoder_state_roundtrip_rank_positive():
    """LoRA-only state round-trip for rank>0."""
    print('=== test_save_load_decoder_state_roundtrip_rank_positive ===')
    # Use q_proj 16x16 = 256 base params vs lora rank 4 → 16*4 + 4*16 = 128;
    # this makes the lora-only-is-smaller assertion meaningful.
    class L(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.q_proj(x)
    wrapped = apply_lora(L(), 4, target_modules=['q_proj'])
    # Run a tiny forward+backward+step so LoRA params change
    x = torch.randn(2, 4, 16)
    out = wrapped(x)
    out.sum().backward()
    opt = torch.optim.SGD(decoder_trainable_params(wrapped, 4), lr=0.1)
    opt.step()
    # Clone to break the reference link between sd and live params.
    sd = {k: v.detach().clone() for k, v in save_decoder_state(wrapped, 4).items()}
    # save should be SMALLER than full model — only LoRA params
    nparams_in_sd = sum(t.numel() for t in sd.values())
    nparams_total = sum(p.numel() for p in wrapped.parameters())
    assert nparams_in_sd < nparams_total, \
        f'rank>0 save should be lora-only ({nparams_in_sd} vs total {nparams_total})'
    # Zero out lora params and reload
    for p in decoder_trainable_params(wrapped, 4):
        p.data.zero_()
    load_decoder_state(wrapped, 4, sd)
    restored = sum(p.abs().sum().item() for p in decoder_trainable_params(wrapped, 4))
    assert restored > 0, 'lora params not restored after load'
    print(f'  lora-only save ({nparams_in_sd} params, total {nparams_total}) round-trips  OK')


def test_init_new_token_embeddings_tied_and_correct_shape():
    print('=== test_init_new_token_embeddings_tied_and_correct_shape ===')
    e, h = init_new_token_embeddings(N_NEW, 2048)
    assert e.weight.shape == (N_NEW, 2048)
    assert h.weight.shape == (N_NEW, 2048)
    assert h.weight is e.weight, 'lm_head must be tied to embed at init'
    e2, h2 = init_new_token_embeddings(0, 2048)
    assert e2 is None and h2 is None, 'n_new_tokens=0 should return (None, None)'
    print('  shapes correct, tying holds, n=0 returns (None, None)  OK')


def test_sample_next_token_bf16_no_nan():
    """bf16 logits through _sample_next_token (temp/top_k/top_p) must not produce NaN.

    Historically torch.multinomial required fp32; modern PyTorch handles bf16,
    but extreme top_p combined with bf16 underflow can still produce NaN.
    """
    from chesslm.utils.eval_utils import _sample_next_token
    print('=== test_sample_next_token_bf16_no_nan ===')
    torch.manual_seed(0)
    V = 1000
    B = 8

    # Build varied logit landscapes: random, heavy-tailed, near-uniform, peaked.
    landscapes = {
        'random':       torch.randn(B, V),
        'peaked':       torch.randn(B, V) * 0.1,  # near-uniform → all softmax probs tiny in bf16
        'sharp':        torch.randn(B, V) * 50,   # extreme range → many -inf-like values
        'one_dominant': torch.full((B, V), -5.0),
    }
    landscapes['one_dominant'][:, 7] = 50.0

    for name, logits_fp32 in landscapes.items():
        logits = logits_fp32.to(torch.bfloat16)
        for temp in (0.0, 0.7, 1.5):
            for top_k in (0, 5, 50):
                for top_p in (1.0, 0.95, 0.5, 1e-3):
                    out = _sample_next_token(logits, temp, top_k, top_p)
                    assert out.shape == (B,), f'{name} t={temp} k={top_k} p={top_p}: shape {out.shape}'
                    assert (out >= 0).all() and (out < V).all(), \
                        f'{name} t={temp} k={top_k} p={top_p}: out-of-range {out}'
                    # All IDs must be finite (integer dtype so just verify no garbage)
                    assert not torch.isnan(out.float()).any(), \
                        f'{name} t={temp} k={top_k} p={top_p}: NaN in output IDs'
    print('  all bf16 (landscape × temp × top_k × top_p) combos return valid IDs  OK')


def test_sample_next_token_greedy_picks_argmax_bf16():
    """temperature=0 must reproduce argmax even in bf16 with closely-spaced logits."""
    from chesslm.utils.eval_utils import _sample_next_token
    print('=== test_sample_next_token_greedy_picks_argmax_bf16 ===')
    V = 100
    logits = torch.zeros(4, V)
    logits[0, 5]  = 1.0
    logits[1, 99] = 0.5
    logits[2, 0]  = 10.0
    logits[3, 42] = 1.0
    out_fp32 = _sample_next_token(logits, 0.0, 0, 1.0)
    out_bf16 = _sample_next_token(logits.to(torch.bfloat16), 0.0, 0, 1.0)
    expected = torch.tensor([5, 99, 0, 42])
    assert (out_fp32 == expected).all(), f'fp32 greedy mismatch: {out_fp32}'
    assert (out_bf16 == expected).all(), f'bf16 greedy mismatch: {out_bf16}'
    print('  greedy argmax stable across fp32/bf16  OK')


# ============================================================================
# GROUP B — per-arch / per-rank tests
# Each test loops over (arch, rank); each iteration builds a fresh model.
# ============================================================================

ARCHS = ('flamingo', 'llava')


def test_init_succeeds_for_all_arch_rank_combinations():
    """All 9 (arch, rank) combinations should build without error."""
    print('=== test_init_succeeds_for_all_arch_rank_combinations ===')
    dev = _device()
    for arch in ARCHS:
        for rank in RANKS:
            model = _build_arch(arch, rank, dev)
            assert model.lora_rank == rank
            assert model.n_new_tokens == N_NEW
            print(f'  {arch:8s} rank={rank:3d}: OK')
            _free(model)


def test_forward_shape_for_all_arch_rank_combinations():
    """forward returns (B, S, frozen_vocab + N_NEW) regardless of arch / rank."""
    print('=== test_forward_shape_for_all_arch_rank_combinations ===')
    dev = _device()
    B, S = 2, 12
    for arch in ARCHS:
        for rank in RANKS:
            model = _build_arch(arch, rank, dev)
            model.eval()
            frozen_vocab = model._base_decoder.config.vocab_size
            ids, enc, attn = _make_inputs(B, S, dev, frozen_vocab)
            with torch.no_grad():
                logits = model(ids, enc, attn)
            expected = (B, S, frozen_vocab + N_NEW)
            assert logits.shape == expected, \
                f'{arch} rank={rank}: got {tuple(logits.shape)}, expected {expected}'
            print(f'  {arch:8s} rank={rank:3d}: shape={tuple(logits.shape)}  OK')
            _free(model)


def test_backward_runs_for_all_arch_rank_combinations():
    """A loss-backward step should not error for any (arch, rank)."""
    print('=== test_backward_runs_for_all_arch_rank_combinations ===')
    dev = _device()
    B, S = 1, 8
    for arch in ARCHS:
        for rank in RANKS:
            model = _build_arch(arch, rank, dev)
            model.train()
            frozen_vocab = model._base_decoder.config.vocab_size
            ids, enc, attn = _make_inputs(B, S, dev, frozen_vocab)
            logits = model(ids, enc, attn)
            loss = logits.float().mean()
            loss.backward()
            # At least one trainable param should have grad
            tp = list(model.trainable_parameters())
            grads = [p.grad for p in tp if p.grad is not None]
            assert grads, f'{arch} rank={rank}: no trainable param received a gradient'
            print(f'  {arch:8s} rank={rank:3d}: backward OK, {len(grads)}/{len(tp)} params with grad')
            _free(model)


def test_gradient_flow_to_expected_params():
    """For each (arch, rank), assert ALL trainable params receive non-None grads
    AND no frozen decoder param receives a gradient."""
    print('=== test_gradient_flow_to_expected_params ===')
    dev = _device()
    B, S = 1, 8
    for arch in ARCHS:
        for rank in RANKS:
            model = _build_arch(arch, rank, dev)
            model.train()
            frozen_vocab = model._base_decoder.config.vocab_size
            ids, enc, attn = _make_inputs(B, S, dev, frozen_vocab)
            logits = model(ids, enc, attn)
            logits.float().mean().backward()

            # All trainable params should have a grad
            missing = []
            for p in model.trainable_parameters():
                if p.grad is None:
                    missing.append(p.shape)
            assert not missing, \
                f'{arch} rank={rank}: {len(missing)} trainable params w/o grad: {missing[:3]}'

            # No FROZEN decoder param should have a grad
            if rank < 0:
                leaks = []
                for n, p in model._base_decoder.named_parameters():
                    if p.grad is not None and p.grad.abs().sum().item() > 0:
                        leaks.append(n)
                assert not leaks, \
                    f'{arch} rank=-1: frozen decoder grads leaked: {leaks[:5]}'

            print(f'  {arch:8s} rank={rank:3d}: all trainable have grads; no leaks  OK')
            _free(model)


def test_param_groups_non_empty_and_optimizer_constructs():
    """param_groups(lr) should never contain an empty 'params' list, and the
    list should construct a valid AdamW optimizer.

    This guards against the LLaVA + lora_rank<0 footgun where the
    decoder param group is [].
    """
    print('=== test_param_groups_non_empty_and_optimizer_constructs ===')
    dev = _device()
    for arch in ARCHS:
        for rank in RANKS:
            model = _build_arch(arch, rank, dev)
            groups = model.param_groups(1e-4)
            for i, g in enumerate(groups):
                assert len(g['params']) > 0, \
                    f'{arch} rank={rank}: param_group[{i}] is EMPTY — AdamW will reject'
            # Construct AdamW
            try:
                opt = torch.optim.AdamW(groups, weight_decay=0.0)
                print(f'  {arch:8s} rank={rank:3d}: {len(groups)} group(s), AdamW ok')
            except Exception as e:
                raise AssertionError(f'{arch} rank={rank}: AdamW failed: {e}')
            _free(model)


def test_param_groups_lr_assignments():
    """Group ordering & LR scaling:
       group 0 = bridge          → lr
       group 1 = decoder (if any)→ lr   (lora_rank>0, fresh adapters)
                                    lr*0.1 (lora_rank=0, backbone unfreeze)
       group N = new_embed       → lr*0.1
    """
    print('=== test_param_groups_lr_assignments ===')
    dev = _device()
    lr = 7.3e-4
    for arch in ARCHS:
        for rank in RANKS:
            model = _build_arch(arch, rank, dev)
            groups = model.param_groups(lr)
            # Group 0: bridge at lr
            assert abs(groups[0]['lr'] - lr) < 1e-12, \
                f'{arch} rank={rank}: bridge group lr={groups[0]["lr"]} ≠ {lr}'

            # Decoder group present iff there are trainable decoder params
            dec_present = (rank >= 0)
            if dec_present:
                expected_dec_lr = lr if rank > 0 else lr * 0.1
                assert abs(groups[1]['lr'] - expected_dec_lr) < 1e-12, \
                    f'{arch} rank={rank}: decoder group lr={groups[1]["lr"]} ≠ {expected_dec_lr}'

            # new_embed group is always last and always at lr*0.1
            assert abs(groups[-1]['lr'] - lr * 0.1) < 1e-12, \
                f'{arch} rank={rank}: new_embed group lr={groups[-1]["lr"]} ≠ {lr*0.1}'
            print(f'  {arch:8s} rank={rank:3d}: bridge={groups[0]["lr"]:.2e}'
                  + (f', dec={groups[1]["lr"]:.2e}' if dec_present else '')
                  + f', new_embed={groups[-1]["lr"]:.2e}  OK')
            _free(model)


def _deepclone_state(sd):
    """Recursive deep clone of a nested state-dict (dict → dict; Tensor → clone)."""
    if isinstance(sd, dict):
        return {k: _deepclone_state(v) for k, v in sd.items()}
    if isinstance(sd, torch.Tensor):
        return sd.detach().clone()
    return sd


def test_save_load_trainable_state_roundtrip():
    """trainable_state_dict → load_trainable_state_dict round-trip preserves the
    forward output bit-exactly (modulo training-mode dropout)."""
    print('=== test_save_load_trainable_state_roundtrip ===')
    dev = _device()
    B, S = 1, 8
    for arch in ARCHS:
        for rank in RANKS:
            model = _build_arch(arch, rank, dev)
            model.eval()
            frozen_vocab = model._base_decoder.config.vocab_size
            ids, enc, attn = _make_inputs(B, S, dev, frozen_vocab)
            with torch.no_grad():
                before = model(ids, enc, attn).clone()

            # Round-trip via in-memory dict. Deep-clone so corruption of live
            # params doesn't propagate to the saved dict (state_dict() returns
            # views/refs by default).
            sd = _deepclone_state(model.trainable_state_dict())
            # Corrupt trainable params, then reload
            for p in model.trainable_parameters():
                p.data.add_(torch.randn_like(p.data))
            with torch.no_grad():
                corrupted = model(ids, enc, attn)
            corruption_diff = (corrupted - before).abs().max().item()
            assert corruption_diff > 1e-3, \
                f'{arch} rank={rank}: corruption failed (model not perturbed)'

            try:
                model.load_trainable_state_dict(sd)
            except KeyError as e:
                raise AssertionError(
                    f'{arch} rank={rank}: load_trainable_state_dict raised '
                    f'KeyError {e!r} — likely missing-decoder-key bug'
                )
            with torch.no_grad():
                after = model(ids, enc, attn)
            diff = (after - before).abs().max().item()
            # bf16 is noisy but reload should be much closer to before than after corruption
            assert diff < corruption_diff / 10, \
                f'{arch} rank={rank}: reload diff={diff:.4f}, corrupt diff={corruption_diff:.4f}'
            print(f'  {arch:8s} rank={rank:3d}: reload diff={diff:.4e}  OK')
            _free(model)


def test_checkpoint_round_trip_via_disk():
    """Realistic checkpoint flow: save state_dict to disk, reload into a freshly
    constructed model, assert outputs match."""
    print('=== test_checkpoint_round_trip_via_disk ===')
    dev = _device()
    B, S = 1, 8
    for arch in ARCHS:
        for rank in RANKS:
            model_a = _build_arch(arch, rank, dev)
            model_a.eval()
            frozen_vocab = model_a._base_decoder.config.vocab_size
            ids, enc, attn = _make_inputs(B, S, dev, frozen_vocab)

            # Perturb trainable params so checkpoint is not just at init
            with torch.no_grad():
                for p in model_a.trainable_parameters():
                    p.data.add_(torch.randn_like(p.data) * 0.01)
            with torch.no_grad():
                out_a = model_a(ids, enc, attn).clone()

            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                ckpt_path = f.name
            try:
                torch.save({'model': model_a.trainable_state_dict()}, ckpt_path)
                _free(model_a)

                model_b = _build_arch(arch, rank, dev)
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model_b.load_trainable_state_dict(ckpt['model'])
                model_b.eval()
                with torch.no_grad():
                    out_b = model_b(ids, enc, attn)
                diff = (out_a - out_b).abs().max().item()
                # bf16 reload via cpu→cuda introduces small noise
                rel = diff / max(out_a.abs().max().item(), 1.0)
                assert rel < 0.05, \
                    f'{arch} rank={rank}: disk round-trip rel diff={rel:.4f}'
                print(f'  {arch:8s} rank={rank:3d}: disk round-trip rel diff={rel:.4e}  OK')
                _free(model_b)
            finally:
                os.unlink(ckpt_path)


def test_new_token_routing_through_new_embed():
    """input_ids >= frozen_vocab should route through model.new_embed, not
    out-of-bounds index into the frozen embed_tokens table."""
    print('=== test_new_token_routing_through_new_embed ===')
    dev = _device()
    B, S = 2, 10
    for arch in ARCHS:
        for rank in RANKS:
            model = _build_arch(arch, rank, dev)
            model.eval()
            frozen_vocab = model._base_decoder.config.vocab_size

            ids_new = torch.randint(frozen_vocab, frozen_vocab + N_NEW, (B, S), device=dev)
            enc = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)

            with torch.no_grad():
                before = model(ids_new, enc).clone()
                saved = model.new_embed.weight.data.clone()
                model.new_embed.weight.data.zero_()
                after = model(ids_new, enc)
                model.new_embed.weight.data.copy_(saved)

            diff = (before - after).abs().max().item()
            assert diff > 1e-3, \
                f'{arch} rank={rank}: zeroing new_embed had no effect — routing broken'
            print(f'  {arch:8s} rank={rank:3d}: routing diff={diff:.4e}  OK')
            _free(model)


def test_weight_tying_preserved_after_from_pretrained_cast():
    """REGRESSION: new_embed.weight and new_lm_head.weight must remain the SAME
    Parameter object after the from_pretrained device/dtype cast.

    This guards against the cpu→cuda `.to()` path that constructs a new
    Parameter and severs tying.
    """
    print('=== test_weight_tying_preserved_after_from_pretrained_cast ===')
    dev = _device()
    for arch in ARCHS:
        # _build_arch follows the same pattern as from_pretrained for the cast.
        model = _build_arch(arch, -1, dev)
        same_param = model.new_lm_head.weight is model.new_embed.weight
        same_storage = (
            model.new_lm_head.weight.data_ptr() == model.new_embed.weight.data_ptr()
        )
        assert same_param, \
            f'{arch}: new_lm_head.weight is NOT the same Parameter as new_embed.weight ' \
            f'after .to(device,dtype) — tying severed.'
        assert same_storage, \
            f'{arch}: storage diverged ({model.new_lm_head.weight.data_ptr()} vs ' \
            f'{model.new_embed.weight.data_ptr()})'
        print(f'  {arch:8s}: tying preserved (same Parameter and same storage)  OK')
        _free(model)


def test_weight_tying_persists_after_optimizer_step():
    """After backward + opt.step, new_embed/new_lm_head must still share weights."""
    print('=== test_weight_tying_persists_after_optimizer_step ===')
    dev = _device()
    B, S = 1, 8
    for arch in ARCHS:
        model = _build_arch(arch, -1, dev)
        model.train()
        frozen_vocab = model._base_decoder.config.vocab_size

        opt = torch.optim.AdamW(model.param_groups(1e-3), weight_decay=0.0)
        ids, enc, attn = _make_inputs(B, S, dev, frozen_vocab)
        labels = torch.randint(frozen_vocab, frozen_vocab + N_NEW, (B, S), device=dev)
        logits = model(ids, enc, attn)
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )
        loss.backward()
        opt.step()

        assert model.new_lm_head.weight is model.new_embed.weight, \
            f'{arch}: tying broken AFTER optimizer step'
        print(f'  {arch:8s}: tying persists after opt.step  OK')
        _free(model)


def test_frozen_decoder_params_bit_exact_after_step_when_rank_negative():
    """rank<0: sampled decoder params must be bit-exactly unchanged after a step."""
    print('=== test_frozen_decoder_params_bit_exact_after_step_when_rank_negative ===')
    dev = _device()
    B, S = 1, 8
    for arch in ARCHS:
        model = _build_arch(arch, -1, dev)
        model.train()
        frozen_vocab = model._base_decoder.config.vocab_size
        # Snapshot a few decoder params
        named = list(model._base_decoder.named_parameters())
        snaps = {n: p.detach().clone() for n, p in named[:6]}

        opt = torch.optim.AdamW(model.param_groups(1e-2), weight_decay=0.0)
        ids, enc, attn = _make_inputs(B, S, dev, frozen_vocab)
        labels = torch.randint(0, frozen_vocab + N_NEW, (B, S), device=dev)
        logits = model(ids, enc, attn)
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )
        loss.backward()
        opt.step()

        cur = dict(model._base_decoder.named_parameters())
        for n, p_old in snaps.items():
            p_new = cur[n]
            assert (p_old == p_new).all().item(), \
                f'{arch} rank=-1: frozen param {n} changed after step!'
        print(f'  {arch:8s}: {len(snaps)} sampled frozen params unchanged  OK')
        _free(model)


def test_lora_adapter_params_change_after_step_when_rank_positive():
    """rank>0: LoRA adapter params should CHANGE after a step; base params should not."""
    print('=== test_lora_adapter_params_change_after_step_when_rank_positive ===')
    dev = _device()
    B, S = 1, 8
    for arch in ARCHS:
        model = _build_arch(arch, 8, dev)
        model.train()
        frozen_vocab = model._base_decoder.config.vocab_size

        # Snapshot LoRA params (the requires_grad=True subset of decoder)
        lora_snaps = {}
        for n, p in model.decoder.named_parameters():
            if p.requires_grad:
                lora_snaps[n] = p.detach().clone()
        assert lora_snaps, f'{arch} rank=8: no LoRA params found'

        # Snapshot a few base params (requires_grad=False)
        base_snaps = {}
        for n, p in model.decoder.named_parameters():
            if not p.requires_grad and 'q_proj' in n:
                base_snaps[n] = p.detach().clone()
                if len(base_snaps) >= 3:
                    break

        opt = torch.optim.AdamW(model.param_groups(1e-2), weight_decay=0.0)
        ids, enc, attn = _make_inputs(B, S, dev, frozen_vocab)
        labels = torch.randint(0, frozen_vocab + N_NEW, (B, S), device=dev)
        logits = model(ids, enc, attn)
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )
        loss.backward()
        opt.step()

        cur = dict(model.decoder.named_parameters())
        changed = sum(
            int(not torch.equal(cur[n], lora_snaps[n])) for n in lora_snaps
        )
        # LoRA B is initialized to zero, so it gets updated; LoRA A starts non-zero and also moves.
        assert changed > 0, f'{arch} rank=8: no LoRA params changed after step'

        unchanged_base = all(torch.equal(cur[n], v) for n, v in base_snaps.items())
        assert unchanged_base, f'{arch} rank=8: base params moved (LoRA should freeze them)'

        print(f'  {arch:8s}: {changed}/{len(lora_snaps)} LoRA params changed; base frozen  OK')
        _free(model)


def test_forward_dtype_matches_input_for_all_arch_rank():
    """logits dtype should match the model's amp_dtype (bf16 in this run)."""
    print('=== test_forward_dtype_matches_input_for_all_arch_rank ===')
    dev = _device()
    B, S = 1, 6
    for arch in ARCHS:
        for rank in RANKS:
            model = _build_arch(arch, rank, dev)
            model.eval()
            frozen_vocab = model._base_decoder.config.vocab_size
            ids, enc, attn = _make_inputs(B, S, dev, frozen_vocab)
            with torch.no_grad():
                logits = model(ids, enc, attn)
            assert logits.dtype == DTYPE, \
                f'{arch} rank={rank}: logits dtype {logits.dtype} != {DTYPE}'
            print(f'  {arch:8s} rank={rank:3d}: dtype OK')
            _free(model)


def test_attention_mask_padding_handled_for_all_arch():
    """Padded sequences (attn_mask with zeros) must run without error and produce
    sensible logits at non-pad positions."""
    print('=== test_attention_mask_padding_handled_for_all_arch ===')
    dev = _device()
    B, S = 2, 12
    for arch in ARCHS:
        model = _build_arch(arch, -1, dev)
        model.eval()
        frozen_vocab = model._base_decoder.config.vocab_size

        ids = torch.randint(0, frozen_vocab, (B, S), device=dev)
        # First row has 4 right-pad tokens
        attn = torch.ones(B, S, dtype=torch.long, device=dev)
        attn[0, -4:] = 0
        enc = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)

        with torch.no_grad():
            logits = model(ids, enc, attn)
        assert logits.shape == (B, S, frozen_vocab + N_NEW)
        # logits at non-pad positions should be finite
        valid = logits[0, :S - 4]
        assert torch.isfinite(valid.float()).all(), f'{arch}: nan/inf in valid positions'
        print(f'  {arch:8s}: padded forward OK')
        _free(model)


# ============================================================================
# GROUP C — cross-arch consistency / spec-level checks
# ============================================================================

def test_protocol_methods_present_on_all_arch():
    """Every arch must expose the ChessLM Protocol surface."""
    print('=== test_protocol_methods_present_on_all_arch ===')
    required = [
        'forward', 'trainable_parameters', 'trainable_state_dict',
        'load_trainable_state_dict', 'get_diagnostics', 'param_groups',
    ]
    for arch in ARCHS:
        cls = _arch_cls(arch)
        for m in required:
            assert hasattr(cls, m), f'{cls.__name__} missing protocol method {m!r}'
    print(f'  all {len(ARCHS)} arch classes implement protocol  OK')


def test_constructors_accept_lora_rank_uniformly():
    """All three constructors must accept lora_rank as a keyword arg."""
    print('=== test_constructors_accept_lora_rank_uniformly ===')
    import inspect
    for arch in ARCHS:
        sig = inspect.signature(_arch_cls(arch).__init__)
        assert 'lora_rank' in sig.parameters, \
            f'{arch}.__init__ missing lora_rank parameter'
    print('  all 3 arch classes accept lora_rank kwarg  OK')


def test_from_pretrained_signatures_accept_lora_rank():
    print('=== test_from_pretrained_signatures_accept_lora_rank ===')
    import inspect
    for arch in ARCHS:
        sig = inspect.signature(_arch_cls(arch).from_pretrained)
        assert 'lora_rank' in sig.parameters, \
            f'{arch}.from_pretrained missing lora_rank parameter'
    print('  all 3 from_pretrained methods accept lora_rank kwarg  OK')


def test_default_lora_rank_per_arch():
    """Default lora_rank per spec: Flamingo=-1 (frozen), LLaVA=0."""
    print('=== test_default_lora_rank_per_arch ===')
    import inspect
    defaults = {}
    for arch in ARCHS:
        sig = inspect.signature(_arch_cls(arch).__init__)
        defaults[arch] = sig.parameters['lora_rank'].default
    assert defaults['flamingo'] == -1, f'flamingo default is {defaults["flamingo"]}, expected -1'
    assert defaults['llava'] == 0, f'llava default is {defaults["llava"]}, expected 0'
    print(f'  defaults: {defaults}  OK')


def test_flamingo_diagnostics_includes_alpha_gates():
    """Flamingo's get_diagnostics should report all 16 alpha_attn and alpha_ffn values."""
    print('=== test_flamingo_diagnostics_includes_alpha_gates ===')
    dev = _device()
    model = _build_arch('flamingo', -1, dev)
    diag = model.get_diagnostics()
    attn_keys = [k for k in diag if k.startswith('alpha_attn/')]
    ffn_keys  = [k for k in diag if k.startswith('alpha_ffn/')]
    assert len(attn_keys) == 16, f'expected 16 alpha_attn entries, got {len(attn_keys)}'
    assert len(ffn_keys) == 16, f'expected 16 alpha_ffn entries, got {len(ffn_keys)}'
    # All alpha values should be in [-1, 1] since they're tanh-transformed
    for k, v in diag.items():
        assert -1.0 <= v <= 1.0, f'{k}={v} out of [-1, 1]'
    print(f'  16 alpha_attn + 16 alpha_ffn entries, all in [-1, 1]  OK')
    _free(model)


def test_llava_diagnostics_empty():
    """LLaVA currently returns empty diagnostics (no LoRA scale tracking)."""
    print('=== test_llava_diagnostics_empty ===')
    dev = _device()
    for arch in ('llava',):
        model = _build_arch(arch, 0, dev)
        diag = model.get_diagnostics()
        assert diag == {}, f'{arch}: expected empty diagnostics, got {list(diag)[:3]}'
        _free(model)
    print('  LLaVA diagnostics empty as documented  OK')


def test_position_ids_offset_contract_llava():
    """For LLaVA, the caller supplies 0-based text position_ids and
    the model internally offsets by N_ENC_SQUARES.

    This codifies the (recently changed) contract so any future regression is
    loud. Test: passing position_ids vs. not passing them should produce the
    same logits for arange(S) text positions."""
    print('=== test_position_ids_offset_contract_llava ===')
    dev = _device()
    B, S = 1, 10
    for arch in ('llava',):
        model = _build_arch(arch, -1, dev)
        model.eval()
        frozen_vocab = model._base_decoder.config.vocab_size
        ids = torch.randint(0, frozen_vocab, (B, S), device=dev)
        enc = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)
        attn = torch.ones(B, S, dtype=torch.long, device=dev)

        with torch.no_grad():
            # Default (no position_ids)
            a = model(ids, enc, attn, position_ids=None)
            # Explicit 0-based positions for text (model offsets internally)
            pos = torch.arange(S, device=dev).unsqueeze(0).expand(B, -1)
            b = model(ids, enc, attn, position_ids=pos)

        diff = (a - b).abs().max().item()
        rel = diff / max(a.abs().max().item(), 1.0)
        assert rel < 0.01, \
            f'{arch}: explicit 0-based position_ids ≠ default; rel diff={rel:.4f} — ' \
            f'caller contract drifted; expected internal offset by N_ENC_SQUARES'
        print(f'  {arch:8s}: 0-based pos_ids matches default (rel diff {rel:.4e})  OK')
        _free(model)


def test_left_pad_position_ids_invariance_per_arch():
    """For each arch, last-position logits under left-padding (with proper
    position_ids from attn_mask cumsum) should match the unpadded version.
    This is the invariant that makes batched generation correct."""
    print('=== test_left_pad_position_ids_invariance_per_arch ===')
    dev = _device()
    L = 8
    for arch in ARCHS:
        model = _build_arch(arch, -1, dev)
        model.eval()
        frozen_vocab = model._base_decoder.config.vocab_size

        real_ids = torch.randint(0, frozen_vocab, (1, L), device=dev)
        enc = torch.randn(1, 16, 64, 1024, dtype=DTYPE, device=dev)

        # Unpadded
        attn_a = torch.ones(1, L, dtype=torch.long, device=dev)
        pos_a = torch.arange(L, device=dev).unsqueeze(0)

        # Left-padded
        P = 5
        pad_id = 0
        padded_ids = torch.full((1, L + P), pad_id, dtype=torch.long, device=dev)
        padded_ids[0, P:] = real_ids[0]
        attn_b = torch.zeros(1, L + P, dtype=torch.long, device=dev)
        attn_b[0, P:] = 1
        pos_b = (attn_b.cumsum(-1) - 1).clamp(min=0)

        with torch.no_grad():
            la = model(real_ids, enc, attn_a, position_ids=pos_a)
            lb = model(padded_ids, enc, attn_b, position_ids=pos_b)

        last_a = la[0, -1].float()
        last_b = lb[0, -1].float()
        abs_diff = (last_a - last_b).abs().max().item()
        rel = abs_diff / max(last_a.abs().max().item(), 1.0)
        # bf16 noise + padding masking can drift up to a few percent
        assert rel < 0.10, \
            f'{arch}: left-pad invariance violated; rel diff={rel:.4f}'
        am_a, am_b = last_a.argmax().item(), last_b.argmax().item()
        # Argmax can differ at the very edge in bf16; check top-5 overlap
        top5_a = set(last_a.topk(5).indices.tolist())
        top5_b = set(last_b.topk(5).indices.tolist())
        overlap = len(top5_a & top5_b)
        assert overlap >= 3 or am_a == am_b, \
            f'{arch}: top-5 overlap only {overlap}; argmax {am_a} vs {am_b}'
        print(f'  {arch:8s}: rel diff={rel:.4f}, top5 overlap={overlap}/5  OK')
        _free(model)


def test_lora_rank_zero_decoder_in_trainable_state_dict():
    """rank=0: decoder state should be in the checkpoint."""
    print('=== test_lora_rank_zero_decoder_in_trainable_state_dict ===')
    dev = _device()
    for arch in ARCHS:
        model = _build_arch(arch, 0, dev)
        sd = model.trainable_state_dict()
        assert 'decoder' in sd, \
            f'{arch} rank=0: "decoder" missing from trainable_state_dict'
        # The decoder state should be sizable (full decoder)
        n_decoder = sum(t.numel() for t in sd['decoder'].values())
        assert n_decoder > 1e9, \
            f'{arch} rank=0: decoder state size {n_decoder} suspiciously small'
        print(f'  {arch:8s}: decoder state present, {n_decoder/1e9:.2f}B params  OK')
        _free(model)


def test_lora_rank_negative_omits_decoder_for_all_arch():
    """rank=-1: "decoder" key MUST be absent from trainable_state_dict for all
    architectures. Decoder is frozen — saving its state wastes GBs per
    checkpoint and is what Flamingo has always done; LLaVA now matches."""
    print('=== test_lora_rank_negative_omits_decoder_for_all_arch ===')
    dev = _device()
    for arch in ARCHS:
        model = _build_arch(arch, -1, dev)
        sd = model.trainable_state_dict()
        assert 'decoder' not in sd, \
            f'{arch} rank=-1: "decoder" key should be absent (frozen → not saved)'
        print(f'  {arch:8s}: rank=-1 omits decoder  OK')
        _free(model)


def test_lora_rank_positive_decoder_state_is_lora_only():
    """rank>0: decoder state in checkpoint should be the LoRA adapters only
    (small), not the full backbone."""
    print('=== test_lora_rank_positive_decoder_state_is_lora_only ===')
    dev = _device()
    for arch in ARCHS:
        model = _build_arch(arch, 8, dev)
        sd = model.trainable_state_dict()
        assert 'decoder' in sd
        n_decoder = sum(t.numel() for t in sd['decoder'].values())
        # LoRA adapters for q/k/v/o on 36 layers at rank 8 ≈ 8 * 4 * 2 * 36 * 2048 ≈ 9.4M
        # Full SmolLM3 3B is way bigger. Should be < 100M.
        assert n_decoder < 100e6, \
            f'{arch} rank=8: decoder state size {n_decoder/1e6:.1f}M — too big for LoRA-only'
        assert n_decoder > 1e6, \
            f'{arch} rank=8: decoder state size {n_decoder/1e6:.1f}M — suspiciously small'
        print(f'  {arch:8s}: rank=8 decoder state = {n_decoder/1e6:.2f}M (LoRA-only)  OK')
        _free(model)


def test_flamingo_peft_compatibility_with_manual_layer_loop():
    """Flamingo iterates decoder layers manually instead of calling decoder().
    Verify that when PEFT wraps the decoder (rank>0), LoRA adapters still fire
    inside the manual loop — i.e., changing LoRA scale changes forward output."""
    print('=== test_flamingo_peft_compatibility_with_manual_layer_loop ===')
    dev = _device()
    B, S = 1, 6
    model = _build_arch('flamingo', 8, dev)
    model.eval()
    frozen_vocab = model._base_decoder.config.vocab_size
    ids, enc, attn = _make_inputs(B, S, dev, frozen_vocab)

    with torch.no_grad():
        before = model(ids, enc, attn).clone()
        # Perturb LoRA *B* weights. PEFT init: A ~ Kaiming, B = 0, so the
        # effective adapter delta ΔW = A @ B starts at zero. Perturbing only
        # A would keep ΔW = A_new @ 0 = 0 → no observable change. Bumping B
        # off zero is what activates the adapter.
        lora_b_params = [p for n, p in model.decoder.named_parameters() if 'lora_B' in n]
        assert lora_b_params, 'no LoRA B params found — PEFT wrap failed'
        saves = [p.data.clone() for p in lora_b_params]
        for p in lora_b_params:
            p.data.add_(torch.randn_like(p.data) * 0.5)
        after = model(ids, enc, attn)
        for p, s in zip(lora_b_params, saves):
            p.data.copy_(s)

    diff = (before - after).abs().max().item()
    assert diff > 1e-3, \
        'Flamingo rank=8: perturbing LoRA B weights had no effect — ' \
        'manual decoder-layer loop is NOT routing through PEFT-patched modules.'
    print(f'  Flamingo rank=8: LoRA perturbation changed output by {diff:.4e}  OK')
    _free(model)


def test_unwrap_decoder_config_access_works_for_all_arch_rank():
    """Verify _base_decoder.config.vocab_size is accessible regardless of LoRA wrap."""
    print('=== test_unwrap_decoder_config_access_works_for_all_arch_rank ===')
    dev = _device()
    for arch in ARCHS:
        for rank in RANKS:
            model = _build_arch(arch, rank, dev)
            base = model._base_decoder
            v = base.config.vocab_size
            assert v > 0
            # Embeddings should be accessible
            assert hasattr(base.model, 'embed_tokens')
            assert hasattr(base, 'lm_head')
            _free(model)
            print(f'  {arch:8s} rank={rank:3d}: base.config.vocab_size={v}  OK')


def test_llava_prefix_attention_mask_extended():
    """LLaVA: when caller passes an attention_mask, it should be extended by 64
    ones on the LEFT to cover prefix tokens. Provide a mask and check that
    forward runs and produces logits over only the S text positions."""
    print('=== test_llava_prefix_attention_mask_extended ===')
    dev = _device()
    B, S = 2, 7
    model = _build_arch('llava', -1, dev)
    model.eval()
    frozen_vocab = model._base_decoder.config.vocab_size
    ids = torch.randint(0, frozen_vocab, (B, S), device=dev)
    enc = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)
    attn = torch.ones(B, S, dtype=torch.long, device=dev)
    with torch.no_grad():
        logits = model(ids, enc, attn)
    # Output sequence dimension should be S (text only), not 64+S.
    assert logits.shape[1] == S, \
        f'LLaVA forward returned shape {tuple(logits.shape)}; expected text length {S}'
    print(f'  LLaVA prefix correctly stripped from output  OK')
    _free(model)


def test_flamingo_x_attn_layers_actually_in_residual_path():
    """Sanity check: perturbing alpha_attn should change the Flamingo forward output."""
    print('=== test_flamingo_x_attn_layers_actually_in_residual_path ===')
    dev = _device()
    B, S = 1, 6
    model = _build_arch('flamingo', -1, dev)
    model.eval()
    frozen_vocab = model._base_decoder.config.vocab_size
    ids = torch.randint(0, frozen_vocab, (B, S), device=dev)
    enc = torch.randn(B, 16, 64, 1024, dtype=DTYPE, device=dev)
    # Close all gates → x-attn contribution is 0
    for layer in model.x_attn_layers:
        layer.alpha_attn.data.fill_(0.0)
        layer.alpha_ffn.data.fill_(0.0)
    with torch.no_grad():
        closed = model(ids, enc).clone()
    # Open all gates → x-attn should contribute
    for layer in model.x_attn_layers:
        layer.alpha_attn.data.fill_(1.0)
        layer.alpha_ffn.data.fill_(1.0)
    with torch.no_grad():
        opened = model(ids, enc)
    diff = (closed - opened).abs().max().item()
    assert diff > 1e-2, f'gate change had no effect; diff={diff}'
    print(f'  gate=0 vs gate=1 differ by {diff:.4f}  OK')
    _free(model)


def test_tokenizer_alignment_with_n_new_tokens():
    """Verify that the tokenizer's added-token range aligns with N_NEW."""
    print('=== test_tokenizer_alignment_with_n_new_tokens ===')
    tok = AutoTokenizer.from_pretrained(DECODER_PATH, local_files_only=True)
    orig = len(tok)
    n_added = tok.add_tokens(ANSWER_SPECIAL_TOKENS, special_tokens=True)
    assert n_added == N_NEW, f'expected {N_NEW} added, got {n_added}'
    cfg = AutoConfig.from_pretrained(DECODER_PATH, local_files_only=True)
    assert orig == cfg.vocab_size, \
        f'tokenizer vocab {orig} != config.vocab_size {cfg.vocab_size} — ' \
        f'frozen_vocab boundary in forward would be wrong'
    print(f'  tokenizer aligned: orig={orig}, added={n_added}, cfg={cfg.vocab_size}  OK')


def test_param_groups_decoder_lr_zero_when_frozen():
    """When a non-empty bridge param group exists and decoder is frozen, the
    optimizer should not contain a 0-element decoder group."""
    print('=== test_param_groups_decoder_lr_zero_when_frozen ===')
    dev = _device()
    for arch in ARCHS:
        model = _build_arch(arch, -1, dev)
        groups = model.param_groups(1e-4)
        for i, g in enumerate(groups):
            n = sum(p.numel() for p in g['params'])
            assert n > 0, f'{arch} rank=-1: group[{i}] has 0 params (would crash AdamW)'
        _free(model)
    print('  no empty groups for any arch + rank=-1  OK')


# ============================================================================
# Runner
# ============================================================================

GROUP_A = [
    test_apply_lora_rank_negative_freezes,
    test_apply_lora_rank_zero_passthrough,
    test_apply_lora_rank_positive_wraps_with_peft,
    test_unwrap_decoder_idempotent_for_plain_decoder,
    test_unwrap_decoder_peels_peft_wrapper,
    test_decoder_trainable_params_partition,
    test_save_load_decoder_state_roundtrip_rank_zero,
    test_save_load_decoder_state_roundtrip_rank_positive,
    test_init_new_token_embeddings_tied_and_correct_shape,
    test_sample_next_token_bf16_no_nan,
    test_sample_next_token_greedy_picks_argmax_bf16,
]

GROUP_B = [
    test_init_succeeds_for_all_arch_rank_combinations,
    test_forward_shape_for_all_arch_rank_combinations,
    test_backward_runs_for_all_arch_rank_combinations,
    test_gradient_flow_to_expected_params,
    test_param_groups_non_empty_and_optimizer_constructs,
    test_param_groups_lr_assignments,
    test_save_load_trainable_state_roundtrip,
    test_checkpoint_round_trip_via_disk,
    test_new_token_routing_through_new_embed,
    test_weight_tying_preserved_after_from_pretrained_cast,
    test_weight_tying_persists_after_optimizer_step,
    test_frozen_decoder_params_bit_exact_after_step_when_rank_negative,
    test_lora_adapter_params_change_after_step_when_rank_positive,
    test_forward_dtype_matches_input_for_all_arch_rank,
    test_attention_mask_padding_handled_for_all_arch,
]

GROUP_C = [
    test_protocol_methods_present_on_all_arch,
    test_constructors_accept_lora_rank_uniformly,
    test_from_pretrained_signatures_accept_lora_rank,
    test_default_lora_rank_per_arch,
    test_flamingo_diagnostics_includes_alpha_gates,
    test_llava_diagnostics_empty,
    test_position_ids_offset_contract_llava,
    test_left_pad_position_ids_invariance_per_arch,
    test_lora_rank_zero_decoder_in_trainable_state_dict,
    test_lora_rank_negative_omits_decoder_for_all_arch,
    test_lora_rank_positive_decoder_state_is_lora_only,
    test_flamingo_peft_compatibility_with_manual_layer_loop,
    test_unwrap_decoder_config_access_works_for_all_arch_rank,
    test_llava_prefix_attention_mask_extended,
    test_flamingo_x_attn_layers_actually_in_residual_path,
    test_tokenizer_alignment_with_n_new_tokens,
    test_param_groups_decoder_lr_zero_when_frozen,
]


def main():
    global DTYPE
    args = sys.argv[1:]
    skip_model = '--no-model' in args
    skip_helpers = '--no-helpers' in args
    args = [a for a in args if a not in ('--no-model', '--no-helpers')]

    if '--dtype' in args:
        i = args.index('--dtype')
        name = args[i + 1]
        if name not in _DTYPE_MAP:
            print(f'Unknown dtype {name!r}; valid: {sorted(set(_DTYPE_MAP))}')
            sys.exit(2)
        DTYPE = _DTYPE_MAP[name]
        del args[i:i + 2]

    target = args[0] if args else None
    print(f'[config] dtype={DTYPE}, skip_model={skip_model}, skip_helpers={skip_helpers}, target={target}')

    tests = []
    if not skip_helpers:
        tests += list(GROUP_A)
    if not skip_model:
        tests += list(GROUP_B) + list(GROUP_C)

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
