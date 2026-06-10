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

# <PIECE_WP> … <PIECE_BK>  (12 tokens)
PIECE_TOKENS: list[str] = [
    "<PIECE_WP>", "<PIECE_WN>", "<PIECE_WB>",
    "<PIECE_WR>", "<PIECE_WQ>", "<PIECE_WK>",
    "<PIECE_BP>", "<PIECE_BN>", "<PIECE_BB>",
    "<PIECE_BR>", "<PIECE_BQ>", "<PIECE_BK>",
]

EMPTY_TOKEN = "<EMPTY>"

# Board-absolute tokens (v2 / v2.1): 64 + 12 + 1 = 77
ANSWER_SPECIAL_TOKENS: list[str] = SQUARE_TOKENS + PIECE_TOKENS + [EMPTY_TOKEN]

# POV-relative tokens (v3): 64 + 12 + 1 = 77
POV_ANSWER_SPECIAL_TOKENS: list[str] = POV_SQUARE_TOKENS + PIECE_TOKENS + [EMPTY_TOKEN]

# (color, piece_type) → token string
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


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def _generate_position_dict(fen: str) -> tuple[dict, dict]:
    """
    Returns:
        square_to_piece : {sq_name: piece_token | EMPTY_TOKEN}
        piece_to_squares : {piece_token: [sq_name, ...]}
    """
    board = chess.Board(fen)

    square_to_piece: dict[str, str] = {}
    for sq in chess.SQUARES:
        sq_name = chess.square_name(sq)
        piece   = board.piece_at(sq)
        square_to_piece[sq_name] = (
            _PIECE_TO_TOKEN[(piece.color, piece.piece_type)] if piece else EMPTY_TOKEN
        )

    piece_to_squares: dict[str, list[str]] = {tok: [] for tok in PIECE_TOKENS}
    for sq_name, tok in square_to_piece.items():
        if tok != EMPTY_TOKEN:
            piece_to_squares[tok].append(sq_name)

    return square_to_piece, piece_to_squares


def _generate_pov_position_dict(fen: str) -> tuple[dict, dict]:
    """POV-relative variant of _generate_position_dict.

    Indexes squares by their LC0 hidden-state index (0–63) rather than board name.
    For white to move: POV index i == board square i.
    For black to move: POV index i == board square i^56 (board is mirrored).

    Returns:
        pov_idx_to_piece  : {pov_idx (int): piece_token | EMPTY_TOKEN}
        piece_to_pov_idxs : {piece_token: [pov_idx, ...]}
    """
    board    = chess.Board(fen)
    is_black = board.turn == chess.BLACK

    pov_idx_to_piece: dict[int, str] = {}
    for pov_idx in range(64):
        board_sq = pov_idx ^ 56 if is_black else pov_idx
        piece    = board.piece_at(board_sq)
        pov_idx_to_piece[pov_idx] = (
            _PIECE_TO_TOKEN[(piece.color, piece.piece_type)] if piece else EMPTY_TOKEN
        )

    piece_to_pov_idxs: dict[str, list[int]] = {tok: [] for tok in PIECE_TOKENS}
    for pov_idx, tok in pov_idx_to_piece.items():
        if tok != EMPTY_TOKEN:
            piece_to_pov_idxs[tok].append(pov_idx)

    return pov_idx_to_piece, piece_to_pov_idxs


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
    (oriented to the side to move), so no post-hoc flip is needed — this matches
    encode_positions(pov=True). The encoder is frozen, so it runs under no_grad.
    """
    with torch.no_grad():
        out = encoder(planes, output_hidden_states=True)
        return torch.stack(out.all_hidden_states, dim=1).to(dtype)
