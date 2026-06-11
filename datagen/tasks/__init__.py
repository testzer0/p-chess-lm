"""Task registry: question_type name -> sample_one(board, frequency, rng) callable.

Adding a task = one import line + one TASKS entry. No `get_task` indirection.
"""
from datagen.tasks import (
    piece_on_diagonal,
    piece_on_file,
    piece_on_rank,
    piece_on_square,
    square_of_piece,
)

TASKS = {
    piece_on_square.NAME:           piece_on_square.sample_one,
    square_of_piece.NAME:           square_of_piece.sample_one,
    piece_on_file.NAME_DIRECT:      piece_on_file.sample_one_direct,
    piece_on_file.NAME_COT:         piece_on_file.sample_one_cot,
    piece_on_rank.NAME_DIRECT:      piece_on_rank.sample_one_direct,
    piece_on_rank.NAME_COT:         piece_on_rank.sample_one_cot,
    piece_on_diagonal.NAME_DIRECT:  piece_on_diagonal.sample_one_direct,
    piece_on_diagonal.NAME_COT:     piece_on_diagonal.sample_one_cot,
}
