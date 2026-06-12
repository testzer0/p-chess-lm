# `datagen/tasks/` — per-task QA generators

Each module owns one Stage-1 task: the entity space it samples over, the
weighter that picks which entity to query, and the prose / parse_tag
rendering for a single fixed entity. Registered in
[`__init__.py:TASKS`](__init__.py).

| Task | Entity | MAX_UNIQUE_QUERIES | Notes |
|---|---|---|---|
| `piece_on_square` | board sq (0..63) | 64 | answer = piece on that square (or `<EMPTY>`) |
| `square_of_piece` | piece species | 12 | answer = list of squares holding that piece |
| `piece_on_file`   | file tuple | 8 | line-task: piece multiset on the file |
| `piece_on_rank`   | rank tuple | 8 | line-task: piece multiset on the rank |
| `piece_on_diagonal` | diagonal tuple | 26 (13 up-right + 13 up-left) | line-task; length-proportional pick |
| `piece_count`     | none | 1 | deterministic — both sides listed |
| `material_count`  | none | 1 | deterministic — both sides listed |

---

## Module contract

Each task module exposes exactly:

```python
NAME: str                                              # e.g. "piece_on_square"
MAX_UNIQUE_QUERIES: int                                # max distinct queries per position

def _choose_entity(board, frequency, rng, exclude: set) -> EntityT:
    """Task-specific weighter over `entities \\ exclude`. Read-only on
    `frequency`; called once per query."""

def _render(entity, board, rng) -> dict:
    """Template + prose rendering for a fixed entity. Returns one record:
        {question, answer, question_type, answer_class}
    """

def sample_n(board, frequency, rng, n: int) -> list[dict]:
    """Standard loop — accumulates `seen`, repeatedly calls _choose_entity
    + _render until n records are collected (capped at MAX_UNIQUE_QUERIES)."""
```

`sample_one(board, frequency, rng)` and
`sample_all(board, frequency, rng)` are **not implemented per task** —
they're synthesized in [`__init__.py`](__init__.py) from `sample_n` +
`MAX_UNIQUE_QUERIES` and attached to each module at import time. Call
sites use them exactly as if they were defined in the module:

```python
from datagen.tasks.piece_on_square import sample_one, sample_all
# both work — provided by __init__.py
```

This removes ~12 lines of identical boilerplate from each task.

### The `frequency` dict

A shared `dict[str, int]` counter the driver maintains per task per split.
After each call to `sample_n`, the driver walks each returned record's
`answer_class` and bumps the counter for every token. The next
`_choose_entity` call reads these counts to shape weights — that's what
keeps the answer-class distribution balanced across many calls.

For tasks whose weighter doesn't read frequency
(`piece_on_file`/`rank`/`diagonal`, `piece_count`, `material_count`), the
arg is accepted but ignored.

### The `seen` set inside `sample_n`

`sample_n(board, frequency, rng, n)` ensures the `n` returned records
query `n` *distinct* entities for that position. The pattern:

```python
def sample_n(board, frequency, rng, n):
    n = min(n, MAX_UNIQUE_QUERIES)
    seen, out = set(), []
    while len(out) < n:
        e = _choose_entity(board, frequency, rng, exclude=seen)
        seen.add(e)
        out.append(_render(e, board, rng))
    return out
```

The per-task weighter re-runs over `entities \ exclude` on every call, so
class balance is preserved across the n draws.

---

## Adding a new task

1. Create `datagen/tasks/<new_task>.py` with:
   - `NAME`, `MAX_UNIQUE_QUERIES`
   - `_choose_entity(board, frequency, rng, exclude)` — weighted pick
   - `_render(entity, board, rng)` — template + prose
   - `sample_n(board, frequency, rng, n)` — the standard loop above
2. Register in `__init__.py`:
   - import the module
   - add to `_TASK_MODULES`
3. If the task needs a non-default grader, add it to
   `utils/eval_utils.py:_GRADERS`.

`sample_one` / `sample_all` are attached automatically by
`__init__.py:_attach_wrappers`.
