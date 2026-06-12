"""Task registry: question_type name -> sample_one(board, frequency, rng) callable.

Adding a task = one import line + one TASKS entry. No `get_task` indirection.
"""
from datagen.tasks import (
    material_count,
    piece_count,
    piece_on_diagonal,
    piece_on_file,
    piece_on_rank,
    piece_on_square,
    square_of_piece,
)

TASKS = {
    piece_on_square.NAME:           piece_on_square.sample_one,
    square_of_piece.NAME:           square_of_piece.sample_one,
    piece_on_file.NAME:             piece_on_file.sample_one,
    piece_on_rank.NAME:             piece_on_rank.sample_one,
    piece_on_diagonal.NAME:         piece_on_diagonal.sample_one,
    piece_count.NAME:               piece_count.sample_one,
    material_count.NAME:            material_count.sample_one,
}
