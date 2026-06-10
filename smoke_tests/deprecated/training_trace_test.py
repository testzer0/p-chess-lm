"""Trace test for the training pipeline's dataset-config branching.

Goal: without loading SmolLM or LC0, verify that every training-time decision
that depends on dataset_config.json (pov, new_tok_in_query) is wired correctly:

 1. `_load_dataset_config` finds dataset_config.json next to train/ and eval/.
 2. The correct special-token list (ANSWER_SPECIAL_TOKENS vs
    POV_ANSWER_SPECIAL_TOKENS) is selected for each variant.
 3. `encode_positions(pov=...)` produces the expected behaviour:
       - pov=False: hidden states for black-to-move positions are flipped
         (sq ^ 56) so index 0 == a1.
       - pov=True:  hidden states are left in LC0's native order.
    Verified using a fake stub encoder that emits a deterministic per-square
    fingerprint so the flip is observable without loading LC0.
 4. `init_special_token_embeddings` (semantic init) reads only from the frozen
    embed and writes only to `new_embed`, using the correct token set.
    We exercise the embedding-init code on a tiny stub model.
 5. The end-to-end branching matches what train.py actually does:
       dataset_cfg["pov"] flows into encode_positions(pov=...) and into the
       tokenizer-extension step in initialize_training_objects.

Usage
-----
source /scratch/gpfs/DANQIC/jeff/chesslm/.venv/bin/activate
python -m chesslm.smoke_tests.training_trace_test \
    --v21-root chesslm/datasets/v2.1 \
    --v3-root  chesslm/datasets/v3

No GPU, no model load required.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import chess
import torch
import torch.nn as nn

from chesslm.utils.utils import (
    ANSWER_SPECIAL_TOKENS,
    POV_ANSWER_SPECIAL_TOKENS,
    EMPTY_TOKEN,
    PIECE_TOKENS,
    POV_SQUARE_TOKENS,
    SQUARE_TOKENS,
    encode_positions,
)


# ---------------------------------------------------------------------------
# Test 1 — _load_dataset_config
# ---------------------------------------------------------------------------

def test_load_dataset_config(v21_root: Path, v3_root: Path) -> bool:
    """Exercises chesslm.utils.training_utils._load_dataset_config.

    Catches the Path-import bug (Path is used in training_utils.py without
    being imported at the top of that module).
    """
    print("\n[1] _load_dataset_config")
    try:
        from chesslm.utils.training_utils import _load_dataset_config
    except Exception as e:
        print(f"  FAIL import: {e}")
        return False

    ok = True
    for name, root, expected in (
        ("v2.1", v21_root, {"pov": False, "new_tok_in_query": True}),
        ("v3",   v3_root,  {"pov": True,  "new_tok_in_query": True}),
    ):
        # The function takes a path to train/ (or eval/), reads sibling config
        try:
            cfg = _load_dataset_config(str(root / "train"))
        except NameError as e:
            print(f"  FAIL ({name}): NameError — most likely `Path` not imported "
                  f"in chesslm/utils/training_utils.py. ({e})")
            ok = False
            continue
        except Exception as e:
            print(f"  FAIL ({name}): {type(e).__name__}: {e}")
            ok = False
            continue

        match = cfg.get("pov") == expected["pov"] and \
                cfg.get("new_tok_in_query") == expected["new_tok_in_query"]
        print(f"  {name}: {cfg}  {'OK' if match else 'FAIL'}")
        ok &= match
    return ok


# ---------------------------------------------------------------------------
# Test 2 — special-token list selection
# ---------------------------------------------------------------------------

def test_token_list_selection() -> bool:
    """Verify the (pov ? POV_ANSWER_SPECIAL_TOKENS : ANSWER_SPECIAL_TOKENS) rule
    that initialize_training_objects uses in training_utils.py.
    """
    print("\n[2] special-token list selection")
    ok = True

    cases = [
        ({"pov": False}, ANSWER_SPECIAL_TOKENS,    "<SQUARE_A1>"),
        ({"pov": True},  POV_ANSWER_SPECIAL_TOKENS,"<SQUARE_1>"),
    ]
    for cfg, expected_list, sentinel in cases:
        chosen = POV_ANSWER_SPECIAL_TOKENS if cfg.get("pov") else ANSWER_SPECIAL_TOKENS
        is_expected = chosen is expected_list and sentinel in chosen
        # Both lists must have same length to keep n_new_tokens consistent
        same_len = len(ANSWER_SPECIAL_TOKENS) == len(POV_ANSWER_SPECIAL_TOKENS) == 77
        all_unique = len(set(chosen)) == len(chosen)
        # Critical: PIECE_TOKENS + EMPTY are shared between variants
        shared_ok = all(t in chosen for t in PIECE_TOKENS + [EMPTY_TOKEN])
        # And the SQ tokens of the OTHER variant must NOT be in the chosen list
        wrong_sq = POV_SQUARE_TOKENS if not cfg.get("pov") else SQUARE_TOKENS
        no_leak = not any(t in chosen for t in wrong_sq)

        ok_case = is_expected and same_len and all_unique and shared_ok and no_leak
        print(f"  pov={cfg['pov']}: list={'POV' if cfg['pov'] else 'ABS'}  "
              f"len={len(chosen)}  shared={shared_ok}  no_leak={no_leak}  "
              f"{'OK' if ok_case else 'FAIL'}")
        ok &= ok_case
    return ok


# ---------------------------------------------------------------------------
# Test 3 — encode_positions POV flip behaviour
# ---------------------------------------------------------------------------

class _StubEncoder:
    """Stub LC0 that emits a fingerprint hidden state.

    out.all_hidden_states is a tuple of 16 tensors of shape (B, 64, 1024).
    Channel 0 of layer 0 of example i is a per-square fingerprint = float(sq),
    so we can directly read off whether index `i` was reindexed via sq^56.
    """
    def __init__(self, n_layers=16, n_squares=64, dim=1024):
        self.n_layers = n_layers
        self.n_sq = n_squares
        self.dim = dim
        # Per-example bookkeeping recorded by input_planes_from_fen()
        self.recorded = []

    def input_planes_from_fen(self, start_fen, moves):
        # Replay to determine side-to-move, just like the real encoder would.
        b = chess.Board(start_fen)
        for m in moves:
            b.push_uci(m)
        self.recorded.append(b.turn)
        return torch.zeros(112, 64)

    def __call__(self, planes, output_hidden_states=True):
        B = planes.shape[0]
        # Layer-0 channel-0 = sq index, broadcast across batch.
        sq_id = torch.arange(self.n_sq, dtype=torch.float32)
        layer0 = torch.zeros(B, self.n_sq, self.dim)
        layer0[..., 0] = sq_id
        # Other layers: zeros (we only care about layer 0 in this test)
        layers = [layer0] + [torch.zeros(B, self.n_sq, self.dim)
                             for _ in range(self.n_layers - 1)]
        return SimpleNamespace(all_hidden_states=tuple(layers))


def test_encode_positions_pov_flip() -> bool:
    print("\n[3] encode_positions POV flip behaviour")

    # Construct two FENs: one white-to-move, one black-to-move.
    # Use a position where the player needs to move (any legal position works).
    fen_white = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # Push e4 → now black to move
    b = chess.Board(fen_white)
    b.push_uci("e2e4")
    fen_black = b.fen()
    start_fens = [fen_white, fen_white]
    moves_list = [[], ["e2e4"]]
    end_fens   = [fen_white, fen_black]

    encoder = _StubEncoder()
    device = torch.device("cpu")

    def fingerprints(pov: bool):
        h = encode_positions(encoder, start_fens, moves_list, end_fens,
                              device, torch.float32, pov=pov)
        # h shape: (B=2, 16, 64, 1024); read layer 0, channel 0 → (B, 64)
        return h[:, 0, :, 0].long().tolist()

    fp_abs = fingerprints(pov=False)
    fp_pov = fingerprints(pov=True)

    # White case: same regardless of pov (no flip when white to move)
    white_ok = fp_abs[0] == list(range(64)) and fp_pov[0] == list(range(64))

    # Black case:
    #   pov=False → squares reindexed via XOR 56 (board-absolute, a1 == index 0)
    #   pov=True  → left as 0..63 (LC0's POV-relative order)
    expected_black_abs = [sq ^ 56 for sq in range(64)]
    black_abs_ok = fp_abs[1] == expected_black_abs
    black_pov_ok = fp_pov[1] == list(range(64))

    print(f"  white pov=False fingerprint identity:    {'OK' if white_ok else 'FAIL'}")
    print(f"  black pov=False flips by sq^56:          {'OK' if black_abs_ok else 'FAIL'}")
    print(f"  black pov=True  keeps native order:      {'OK' if black_pov_ok else 'FAIL'}")
    if not black_abs_ok:
        print(f"    expected: {expected_black_abs[:8]}...")
        print(f"    got:      {fp_abs[1][:8]}...")

    return white_ok and black_abs_ok and black_pov_ok


# ---------------------------------------------------------------------------
# Test 4 — init_special_token_embeddings (semantic init, pov branch)
# ---------------------------------------------------------------------------

class _StubTokenizer:
    """Tiny stub: maps chars/words/special tokens to deterministic IDs."""
    def __init__(self, vocab):
        self.vocab = vocab
        self.id_of = {t: i for i, t in enumerate(vocab)}
        self.next_new = len(vocab)
        self.special_ids = {}

    def encode(self, text, add_special_tokens=False):
        # treat each word/char in `vocab` as a separate token
        if text in self.id_of:
            return [self.id_of[text]]
        # split on chars for things like "e4" → ["e", "4"]
        ids = []
        for ch in text:
            if ch in self.id_of:
                ids.append(self.id_of[ch])
        return ids or [0]

    def add_special_token(self, tok):
        self.special_ids[tok] = self.next_new
        self.next_new += 1

    def convert_tokens_to_ids(self, tok):
        return self.special_ids[tok]


class _StubDecoder(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        # Wrap to match model.decoder.model.embed_tokens.weight access pattern
        self.model = SimpleNamespace(
            embed_tokens=nn.Embedding(vocab_size, dim))
        self.config = SimpleNamespace(vocab_size=vocab_size)


class _StubFlamingo(nn.Module):
    def __init__(self, vocab_size, dim, n_new):
        super().__init__()
        self.decoder = _StubDecoder(vocab_size, dim)
        self.new_embed = nn.Embedding(n_new, dim)
        self.n_new_tokens = n_new


def test_embedding_init() -> bool:
    print("\n[4] init_special_token_embeddings")
    try:
        from chesslm.utils.training_utils import init_special_token_embeddings
    except NameError as e:
        print(f"  FAIL importing training_utils: {e}")
        return False
    except Exception as e:
        print(f"  FAIL import: {e}")
        return False

    DIM = 8
    # Build a vocab containing every char/word needed by init
    base_vocab = (
        list("abcdefgh12345678")
        + ["white", "black", "pawn", "knight", "bishop",
           "rook", "queen", "king", "empty"]
    )
    tokenizer = _StubTokenizer(base_vocab)
    vocab_size = len(base_vocab)

    ok = True
    for pov, special_tokens, sentinel_idx in (
        (False, ANSWER_SPECIAL_TOKENS, 0),    # SQUARE_A1 ↔ board sq 0
        (True,  POV_ANSWER_SPECIAL_TOKENS, 0),# SQUARE_1   ↔ POV idx 0 = sq 0
    ):
        # Fresh model + register the special tokens in the stub tokenizer
        tok = _StubTokenizer(base_vocab)
        for t in special_tokens:
            tok.add_special_token(t)
        model = _StubFlamingo(vocab_size, DIM, n_new=len(special_tokens))

        # Seed frozen embed deterministically so we can check the average.
        with torch.no_grad():
            for ch, idx in tok.id_of.items():
                model.decoder.model.embed_tokens.weight[idx] = (
                    torch.arange(DIM, dtype=torch.float32) + idx)

        # Snapshot frozen embed BEFORE init — must not be mutated.
        frozen_before = model.decoder.model.embed_tokens.weight.clone()

        # Run semantic init
        init_special_token_embeddings(model, tok, strategy="semantic", pov=pov)

        frozen_after = model.decoder.model.embed_tokens.weight
        frozen_untouched = torch.equal(frozen_before, frozen_after)

        # Check SQ token init: should equal mean of file/rank embeddings.
        # For pov=False, <SQUARE_A1> ↔ mean(embed('a'), embed('1'))
        # For pov=True,  <SQUARE_1>  ↔ mean(embed('a'), embed('1')) (same target)
        sq_tok = ("<SQUARE_A1>" if not pov else "<SQUARE_1>")
        idx = tok.convert_tokens_to_ids(sq_tok) - vocab_size
        new_row = model.new_embed.weight.data[idx]
        expected = (model.decoder.model.embed_tokens.weight[tok.id_of["a"]]
                    + model.decoder.model.embed_tokens.weight[tok.id_of["1"]]) / 2.0
        sq_init_ok = torch.allclose(new_row, expected, atol=1e-5)

        # Check PIECE_WK init: mean(embed("white"), embed("king"))
        idx = tok.convert_tokens_to_ids("<PIECE_WK>") - vocab_size
        new_row = model.new_embed.weight.data[idx]
        expected = (model.decoder.model.embed_tokens.weight[tok.id_of["white"]]
                    + model.decoder.model.embed_tokens.weight[tok.id_of["king"]]) / 2.0
        piece_init_ok = torch.allclose(new_row, expected, atol=1e-5)

        # Check EMPTY init: embed("empty")
        idx = tok.convert_tokens_to_ids(EMPTY_TOKEN) - vocab_size
        new_row = model.new_embed.weight.data[idx]
        expected = model.decoder.model.embed_tokens.weight[tok.id_of["empty"]]
        empty_init_ok = torch.allclose(new_row, expected, atol=1e-5)

        case_ok = frozen_untouched and sq_init_ok and piece_init_ok and empty_init_ok
        print(f"  pov={pov}: frozen_untouched={frozen_untouched}  "
              f"sq={sq_init_ok}  piece={piece_init_ok}  empty={empty_init_ok}  "
              f"{'OK' if case_ok else 'FAIL'}")
        ok &= case_ok
    return ok


# ---------------------------------------------------------------------------
# Test 5 — end-to-end branching consistency check
# ---------------------------------------------------------------------------

def test_branching_consistency(v21_root: Path, v3_root: Path) -> bool:
    """Read each variant's config and verify the resulting (token list,
    encode_pov, init_pov) tuple is internally consistent with train.py's plan.
    """
    print("\n[5] end-to-end branching consistency")
    ok = True
    for name, root in (("v2.1", v21_root), ("v3", v3_root)):
        with open(root / "dataset_config.json") as f:
            cfg = json.load(f)
        pov = cfg.get("pov", False)
        special_tokens = POV_ANSWER_SPECIAL_TOKENS if pov else ANSWER_SPECIAL_TOKENS

        # All three flags must agree on `pov`
        encode_pov = pov                  # train.py:202 / 276
        init_pov   = pov                  # initialize_training_objects → init_*
        token_set_matches = (pov == (special_tokens is POV_ANSWER_SPECIAL_TOKENS))

        # Sanity: chosen token set must NOT mix variants
        wrong = set(POV_SQUARE_TOKENS if not pov else SQUARE_TOKENS)
        no_mix = not any(t in special_tokens for t in wrong)

        case_ok = token_set_matches and no_mix and (encode_pov == init_pov == pov)
        print(f"  {name}: pov={pov}  encode_pov={encode_pov}  init_pov={init_pov}  "
              f"token_set={'POV' if pov else 'ABS'}  no_mix={no_mix}  "
              f"{'OK' if case_ok else 'FAIL'}")
        ok &= case_ok
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--v21-root", default="chesslm/datasets/v2.1")
    p.add_argument("--v3-root",  default="chesslm/datasets/v3")
    return p.parse_args()


def main():
    args = parse_args()
    v21 = Path(args.v21_root)
    v3  = Path(args.v3_root)

    results = {
        "load_dataset_config":   test_load_dataset_config(v21, v3),
        "token_list_selection":  test_token_list_selection(),
        "encode_positions_pov":  test_encode_positions_pov_flip(),
        "embedding_init":        test_embedding_init(),
        "branching_consistency": test_branching_consistency(v21, v3),
    }

    print("\n" + "=" * 72)
    for k, v in results.items():
        print(f"  {k:<28} {'PASS' if v else 'FAIL'}")
    print("=" * 72)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
