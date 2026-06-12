"""BoardRepr — single position primitive shared by datagen and eval.

Hides the absolute / POV coordinate-system distinction behind one API.
Internal keys are always python-chess board square indices (0..63, a1=0);
all POV rendering happens at the boundary via `sq_tok`.

Construction-time `pov` flag selects:
    pov=False  -> <SQUARE_A1>..<SQUARE_H8>, <PIECE_WP>/<PIECE_BP>
    pov=True   -> <SQUARE_1>..<SQUARE_64>,  <PIECE_MP>/<PIECE_OP>
                  (with board_sq ^ 56 flip when side-to-move is black)

Callers should NEVER branch on `pov` themselves — go through this class.
"""
import chess

from utils.utils import (
    EMPTY_TOKEN,
    PIECE_TOKENS,
    POV_PIECE_TOKENS,
    POV_SQUARE_TOKENS,
    SQUARE_TOKENS,
    _PIECE_TO_POV_TOKEN,
    _PIECE_TO_TOKEN,
)


class BoardRepr:
    def __init__(self, fen: str, pov: bool):
        self.pov = pov
        board = chess.Board(fen)
        self._is_black_pov = pov and (board.turn == chess.BLACK)
        self._stm = board.turn

        self.piece_tokens = POV_PIECE_TOKENS if pov else PIECE_TOKENS

        self._sq_to_piece: dict[int, str] = {}
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                self._sq_to_piece[sq] = EMPTY_TOKEN
            elif pov:
                self._sq_to_piece[sq] = _PIECE_TO_POV_TOKEN[
                    (piece.color == self._stm, piece.piece_type)
                ]
            else:
                self._sq_to_piece[sq] = _PIECE_TO_TOKEN[
                    (piece.color, piece.piece_type)
                ]

        self._piece_to_sqs: dict[str, list[int]] = {
            tok: [] for tok in self.piece_tokens
        }
        # Iterate in POV display order when pov=True so list order matches
        # the order sq_tok would render. For abs, board sq order (0..63) is
        # already the canonical display order.
        order = (
            [sq ^ 56 for sq in range(64)] if self._is_black_pov
            else list(range(64))
        )
        for board_sq in order:
            tok = self._sq_to_piece[board_sq]
            if tok != EMPTY_TOKEN:
                self._piece_to_sqs[tok].append(board_sq)

    @classmethod
    def from_fen(cls, fen: str, pov: bool) -> "BoardRepr":
        return cls(fen, pov)

    # ------------------------------------------------------------------
    # Position access (board_sq is python-chess 0..63, a1=0)
    # ------------------------------------------------------------------

    def piece_at(self, board_sq: int) -> str:
        return self._sq_to_piece[board_sq]

    def squares_with(self, piece_tok: str) -> list[int]:
        return self._piece_to_sqs[piece_tok]

    # ------------------------------------------------------------------
    # Token rendering
    # ------------------------------------------------------------------

    def sq_tok(self, board_sq: int) -> str:
        if self.pov:
            pov_idx = board_sq ^ 56 if self._is_black_pov else board_sq
            return POV_SQUARE_TOKENS[pov_idx]
        return SQUARE_TOKENS[board_sq]

    # ------------------------------------------------------------------
    # Whole-board dicts (subsumes utils.utils._generate_*position_dict)
    # ------------------------------------------------------------------

    def square_to_piece(self) -> dict[str, str]:
        """{sq_tok: piece_tok | EMPTY_TOKEN}, iteration order = display order."""
        order = (
            [sq ^ 56 for sq in range(64)] if self._is_black_pov
            else list(range(64))
        )
        return {self.sq_tok(sq): self._sq_to_piece[sq] for sq in order}

    def piece_to_squares(self) -> dict[str, list[str]]:
        """{piece_tok: [sq_tok, ...]}, square lists in display order."""
        return {
            tok: [self.sq_tok(sq) for sq in self._piece_to_sqs[tok]]
            for tok in self.piece_tokens
        }

    # ------------------------------------------------------------------
    # Geometric entity enumeration. Each entity is a tuple of board square
    # indices ordered to match canonical render order under sq_tok:
    #   pov=False : board-absolute, bottom-up   (a-file = a1..a8, etc.)
    #   pov=True  : POV bottom-up from side-to-move's view  (POV col c reads
    #               POV idx [c, c+8, ..., c+56], regardless of color)
    # FEN-independent geometry (POV flip handled here, not in callers).
    # ------------------------------------------------------------------

    def _pov_idx_to_board(self, pov_idx: int) -> int:
        return pov_idx ^ 56 if self._is_black_pov else pov_idx

    def files(self) -> list[tuple[int, ...]]:
        """8 files, each enumerated bottom-up in render order."""
        if self.pov:
            return [
                tuple(self._pov_idx_to_board(c + r * 8) for r in range(8))
                for c in range(8)
            ]
        return [tuple(chess.square(f, r) for r in range(8)) for f in range(8)]

    def ranks(self) -> list[tuple[int, ...]]:
        """8 ranks, each enumerated left-to-right in render order."""
        if self.pov:
            return [
                tuple(self._pov_idx_to_board(c + r * 8) for c in range(8))
                for r in range(8)
            ]
        return [tuple(chess.square(f, r) for f in range(8)) for r in range(8)]

    def diagonals(self) -> list[tuple[int, ...]]:
        """26 diagonals (13 up-right + 13 up-left), length >= 2.

        All enumerated lowest-rank-first, so both groups walk bottom-up
        from the player's view; they differ only by horizontal direction:

        Up-right (rank - file = c,  c in [-6, 6]):
            e.g. a1->h8 main, a2->g8, b1->h7.
        Up-left  (file + rank = c', c' in [1, 13]):
            e.g. h1->a8 anti, g1->a7, h2->b8.

        POV mode applies the LC0 vertical mirror to the board squares; the
        rank/file used for the c constants is POV's, so for black-to-move
        the "POV up-right" set maps to board squares that descend in
        absolute rank but still read up-right from the player.
        """
        if self.pov:
            up_right = [
                tuple(
                    self._pov_idx_to_board(f + (f + c) * 8)
                    for f in range(max(0, -c), min(8, 8 - c))
                )
                for c in range(-6, 7)
            ]
            up_left = [
                tuple(
                    self._pov_idx_to_board(f + (cp - f) * 8)
                    for f in range(min(7, cp), max(0, cp - 7) - 1, -1)
                )
                for cp in range(1, 14)
            ]
            return up_right + up_left

        up_right = [
            tuple(
                chess.square(f, f + c)
                for f in range(max(0, -c), min(8, 8 - c))
            )
            for c in range(-6, 7)
        ]
        up_left = [
            tuple(
                chess.square(f, cp - f)
                for f in range(min(7, cp), max(0, cp - 7) - 1, -1)
            )
            for cp in range(1, 14)
        ]
        return up_right + up_left
