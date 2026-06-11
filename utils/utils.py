"""Shared constants and utilities for data generation, training, and eval."""
import chess
import torch

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

# <SQUARE_A1> … <SQUARE_H8>  (64 tokens, indexed by chess.SQUARES order a1=0)
SQUARE_TOKENS: list[str] = [
    f"<SQUARE_{chess.square_name(sq).upper()}>"
    for sq in chess.SQUARES
]

# <SQUARE_1> … <SQUARE_64>  (64 POV-relative tokens, index i = LC0 hidden state i)
# For white to move: SQUARE_i corresponds to board square i (a1=1, b1=2, …)
# For black to move: SQUARE_i corresponds to board square i^56 (a8=1, b8=2, …)
POV_SQUARE_TOKENS: list[str] = [f"<SQUARE_{i + 1}>" for i in range(64)]

# <PIECE_WP> … <PIECE_BK>  (12 absolute tokens, used with pov=False)
PIECE_TOKENS: list[str] = [
    "<PIECE_WP>", "<PIECE_WN>", "<PIECE_WB>",
    "<PIECE_WR>", "<PIECE_WQ>", "<PIECE_WK>",
    "<PIECE_BP>", "<PIECE_BN>", "<PIECE_BB>",
    "<PIECE_BR>", "<PIECE_BQ>", "<PIECE_BK>",
]

# <PIECE_MP> … <PIECE_OK>  (12 POV tokens, used with pov=True)
# M = mine (same color as side-to-move), O = opponent's.
# Piece-letter ordering matches PIECE_TOKENS (P/N/B/R/Q/K) so e.g.
# POV_PIECE_TOKENS[0] = "<PIECE_MP>" corresponds to LC0 input plane 0
# (my pawns) and POV_PIECE_TOKENS[6] = "<PIECE_OP>" corresponds to plane 6
# (opponent's pawns).
POV_PIECE_TOKENS: list[str] = [
    "<PIECE_MP>", "<PIECE_MN>", "<PIECE_MB>",
    "<PIECE_MR>", "<PIECE_MQ>", "<PIECE_MK>",
    "<PIECE_OP>", "<PIECE_ON>", "<PIECE_OB>",
    "<PIECE_OR>", "<PIECE_OQ>", "<PIECE_OK>",
]

EMPTY_TOKEN = "<EMPTY>"

# Board-absolute tokens (v2 / v2.1): 64 + 12 + 1 = 77
ANSWER_SPECIAL_TOKENS: list[str] = SQUARE_TOKENS + PIECE_TOKENS + [EMPTY_TOKEN]

# POV-relative tokens (v3.1+): 64 + 12 + 1 = 77, fully disjoint from
# ANSWER_SPECIAL_TOKENS (POV squares + POV pieces).
POV_ANSWER_SPECIAL_TOKENS: list[str] = POV_SQUARE_TOKENS + POV_PIECE_TOKENS + [EMPTY_TOKEN]

# (color, piece_type) → token string  (absolute)
_PIECE_TO_TOKEN: dict[tuple[bool, int], str] = {
    (chess.WHITE, chess.PAWN):   "<PIECE_WP>",
    (chess.WHITE, chess.KNIGHT): "<PIECE_WN>",
    (chess.WHITE, chess.BISHOP): "<PIECE_WB>",
    (chess.WHITE, chess.ROOK):   "<PIECE_WR>",
    (chess.WHITE, chess.QUEEN):  "<PIECE_WQ>",
    (chess.WHITE, chess.KING):   "<PIECE_WK>",
    (chess.BLACK, chess.PAWN):   "<PIECE_BP>",
    (chess.BLACK, chess.KNIGHT): "<PIECE_BN>",
    (chess.BLACK, chess.BISHOP): "<PIECE_BB>",
    (chess.BLACK, chess.ROOK):   "<PIECE_BR>",
    (chess.BLACK, chess.QUEEN):  "<PIECE_BQ>",
    (chess.BLACK, chess.KING):   "<PIECE_BK>",
}

# (is_mine, piece_type) → token string  (POV)
# is_mine = (piece.color == board.turn)
_PIECE_TO_POV_TOKEN: dict[tuple[bool, int], str] = {
    (True,  chess.PAWN):   "<PIECE_MP>",
    (True,  chess.KNIGHT): "<PIECE_MN>",
    (True,  chess.BISHOP): "<PIECE_MB>",
    (True,  chess.ROOK):   "<PIECE_MR>",
    (True,  chess.QUEEN):  "<PIECE_MQ>",
    (True,  chess.KING):   "<PIECE_MK>",
    (False, chess.PAWN):   "<PIECE_OP>",
    (False, chess.KNIGHT): "<PIECE_ON>",
    (False, chess.BISHOP): "<PIECE_OB>",
    (False, chess.ROOK):   "<PIECE_OR>",
    (False, chess.QUEEN):  "<PIECE_OQ>",
    (False, chess.KING):   "<PIECE_OK>",
}


# ---------------------------------------------------------------------------
# Chat prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are ChessLM, an AI assistant with deep chess knowledge. "
    "You can analyze board positions and identify piece locations precisely. "
    "/no_think /system_override"
)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

_BLACK_IDX = torch.tensor([sq ^ 56 for sq in range(64)])


@torch.no_grad()
def encode_positions(
    encoder,
    start_fens: list[str],
    moves_list: list[list[str]],
    end_fens: list[str],
    device: torch.device,
    dtype: torch.dtype,
    pov: bool = False,
) -> torch.Tensor:
    """Returns encoder hidden states (B, 16, 64, 1024).

    pov=False (default): un-flip black positions so index 0 always = a1
                         (board-absolute, matches SQUARE_TOKENS).
    pov=True:            leave hidden states in LC0's native order so index i
                         always = "my bottom-left + i" from current player's POV
                         (matches POV_SQUARE_TOKENS).
    """
    planes = torch.stack([
        encoder.input_planes_from_fen(sf, mv)
        for sf, mv in zip(start_fens, moves_list)
    ]).to(device)

    out = encoder(planes, output_hidden_states=True)
    hidden = torch.stack(out.all_hidden_states, dim=1).to(dtype)  # (B, 16, 64, 1024)

    if not pov:
        black_idx = _BLACK_IDX.to(device)
        for i, (sf, mv) in enumerate(zip(start_fens, moves_list)):
            board = chess.Board(sf)
            for m in mv:
                board.push_uci(m)
            if board.turn == chess.BLACK:
                hidden[i] = hidden[i, :, black_idx, :]

    return hidden


def encode_planes(encoder, planes: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Run the lc0 encoder on pre-built planes -> hidden states (B, 16, 64, 1024).

    The planes come from our lc0_planes builder and are already POV-relative
    (oriented to the side to move), so no post-hoc flip is needed. The encoder is
    frozen, so it runs under no_grad.

    Two encoder interfaces are supported. Our lc0 (src/lc0_torch) returns only
    last_hidden_state, so we capture each layer by wrapping ``_forward_encoder``
    (the input-embedding output + the 15 encoder-layer outputs = 16 states, same
    convention as chess_lm_base). An encoder that natively exposes per-layer
    states via ``output_hidden_states`` is used directly.
    """
    with torch.no_grad():
        if hasattr(encoder, "_forward_encoder"):
            captured: list[torch.Tensor] = []
            orig = encoder._forward_encoder

            def wrapped(x, layer):
                if not captured:
                    captured.append(x)          # input-embedding output (state 0)
                out = orig(x, layer)
                captured.append(out)            # encoder-layer outputs (states 1..15)
                return out

            encoder._forward_encoder = wrapped
            try:
                encoder(input_planes=planes, return_dict=True)
            finally:
                del encoder._forward_encoder    # restore the bound method
            return torch.stack(captured, dim=1).to(dtype)

        out = encoder(planes, output_hidden_states=True)
        return torch.stack(out.all_hidden_states, dim=1).to(dtype)
