"""Optional chess special-token support (all the token-adding machinery).

The repo's chess tokenizers already contain the answer tokens the data uses, so
by default the trainer adds nothing (n_new_tokens=0). When a tokenizer lacks
them, set ``add_special_tokens: true`` in the config:

  * maybe_add_special_tokens     — adds the answer-token set to the tokenizer
                                   *before* the model is built (so n_new_tokens
                                   is known).
  * maybe_init_special_token_embeddings — semantically initializes the new
                                   embeddings *after* the model is built.

Both are no-ops when the flag is off.
"""
import chess
import torch

from chesslm.models.base import unwrap_decoder
from chesslm.utils.utils import (
    POV_ANSWER_SPECIAL_TOKENS,
    EMPTY_TOKEN,
    POV_SQUARE_TOKENS,
    _PIECE_TO_TOKEN,
)

_COLOR_WORDS = {chess.WHITE: "white", chess.BLACK: "black"}
_PIECE_WORDS = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}


def maybe_add_special_tokens(tokenizer, args) -> int:
    """Add the POV chess answer tokens to ``tokenizer`` iff ``args.add_special_tokens``.

    Returns the number of tokens added (0 when the flag is off). Call this BEFORE
    building the model so the model can size its new-token embeddings.
    """
    if not getattr(args, "add_special_tokens", False):
        return 0
    before = len(tokenizer)
    tokenizer.add_tokens(POV_ANSWER_SPECIAL_TOKENS, special_tokens=True)
    return len(tokenizer) - before


def _mean_embedding(embed_weight: torch.Tensor, tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return embed_weight[ids].float().mean(dim=0)


def maybe_init_special_token_embeddings(model, tokenizer, args) -> None:
    """Semantically initialize the newly added token embeddings.

    No-op when the flag is off, no tokens were added, or embed_init='random'.
    Reads from the frozen pretrained embeddings; never modifies them.

    semantic init:
      POV SQUARE tokens ← mean(file_char, rank_char) of board square i (POV index
        i maps to board square i, the white-POV correspondence)
      piece tokens      ← mean(color_word, piece_word)
      EMPTY token       ← embedding of 'empty'
    """
    if (not getattr(args, "add_special_tokens", False)
            or model.n_new_tokens == 0
            or getattr(args, "embed_init", "semantic") == "random"):
        return

    base_decoder = unwrap_decoder(model.decoder)
    frozen_w = base_decoder.model.embed_tokens.weight.data
    new_emb_w = model.new_embed.weight.data
    frozen_vocab = base_decoder.config.vocab_size

    sq_semantic = {}
    for sq in chess.SQUARES:
        name = chess.square_name(sq)
        sq_semantic[sq] = (
            _mean_embedding(frozen_w, tokenizer, name[0])
            + _mean_embedding(frozen_w, tokenizer, name[1])
        ) / 2.0

    for i, tok in enumerate(POV_SQUARE_TOKENS):
        idx = tokenizer.convert_tokens_to_ids(tok) - frozen_vocab
        new_emb_w[idx] = sq_semantic[i].to(new_emb_w.dtype)

    for (color, ptype), tok in _PIECE_TO_TOKEN.items():
        avg = (
            _mean_embedding(frozen_w, tokenizer, _COLOR_WORDS[color])
            + _mean_embedding(frozen_w, tokenizer, _PIECE_WORDS[ptype])
        ) / 2.0
        idx = tokenizer.convert_tokens_to_ids(tok) - frozen_vocab
        new_emb_w[idx] = avg.to(new_emb_w.dtype)

    empty_idx = tokenizer.convert_tokens_to_ids(EMPTY_TOKEN) - frozen_vocab
    new_emb_w[empty_idx] = _mean_embedding(frozen_w, tokenizer, "empty").to(new_emb_w.dtype)
