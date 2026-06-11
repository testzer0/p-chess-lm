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
    # Geometric entity enumeration (board-absolute square indices)
    # POV rendering is downstream via sq_tok. FEN-independent.
    # ------------------------------------------------------------------

    def files(self) -> list[tuple[int, ...]]:
        """8 files, each (rank-1 .. rank-8) for that file."""
        return [tuple(chess.square(f, r) for r in range(8)) for f in range(8)]

    def ranks(self) -> list[tuple[int, ...]]:
        """8 ranks, each (file-a .. file-h) for that rank."""
        return [tuple(chess.square(f, r) for f in range(8)) for r in range(8)]

    def diagonals(self) -> list[tuple[int, ...]]:
        """13 rising diagonals (rank - file = c, c in [-6, 6], length >= 2).

        Falling-axis diagonals are intentionally omitted — see the locked
        decision in plans/merge_data_pipelines.md.
        """
        return [
            tuple(chess.square(f, f + c) for f in range(8) if 0 <= f + c < 8)
            for c in range(-6, 7)
        ]
