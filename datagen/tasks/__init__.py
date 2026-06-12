"""Task registry: question_type name -> task module.

Each task module exposes:
    NAME: str
    MAX_UNIQUE_QUERIES: int
    sample_n(board, frequency, rng, n) -> list[dict]
    _choose_entity / _render          — internal helpers

`sample_one` and `sample_all` are trivial wrappers over `sample_n` (identical
across every task), so they're synthesized here and attached to each module
at import time rather than duplicated in seven files. Existing call sites
(`module.sample_one(...)`, `module.sample_all(...)`) keep working unchanged.
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

_TASK_MODULES = [
    piece_on_square,
    square_of_piece,
    piece_on_file,
    piece_on_rank,
    piece_on_diagonal,
    piece_count,
    material_count,
]


def _attach_wrappers(module) -> None:
    """Synthesize sample_one + sample_all from the module's sample_n + MAX_UNIQUE_QUERIES."""
    def sample_one(board, frequency, rng):
        return module.sample_n(board, frequency, rng, 1)[0]

    def sample_all(board, frequency, rng):
        return module.sample_n(board, frequency, rng, module.MAX_UNIQUE_QUERIES)

    module.sample_one = sample_one
    module.sample_all = sample_all


for _m in _TASK_MODULES:
    _attach_wrappers(_m)


TASKS = {m.NAME: m for m in _TASK_MODULES}
