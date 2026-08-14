"""Grid traversal helpers.

Shared by the level loader (which uses reachability to reject unsolvable
procedural boards) and the state encoder (which uses free-space counts as the
features that let the agent see a pocket before it seals itself in).
"""
from collections import deque
from typing import Iterable

from .types import DELTA, Point


def neighbours(cell: Point, cols: int, rows: int) -> Iterable[Point]:
    """Yield the in-bounds orthogonal neighbours of a cell."""
    for dx, dy in DELTA.values():
        nx, ny = cell.x + dx, cell.y + dy
        if 0 <= nx < cols and 0 <= ny < rows:
            yield Point(nx, ny)


def flood_fill(start: Point, blocked, cols: int, rows: int, limit: int | None = None) -> set[Point]:
    """Return the cells reachable from `start` without entering `blocked`.

    `start` itself is excluded from the result if it is blocked. `limit` caps the
    search, which matters in the state encoder: it runs on every frame of
    training, and the agent only needs to know whether space is *ample*, not its
    exact size.
    """
    if start in blocked or not (0 <= start.x < cols and 0 <= start.y < rows):
        return set()

    seen = {start}
    if limit is not None and len(seen) >= limit:
        return seen

    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbour in neighbours(cell, cols, rows):
            if neighbour not in seen and neighbour not in blocked:
                seen.add(neighbour)
                # Check on insertion, not once per pop: a single pop can add four
                # cells, so a per-pop check overshoots `limit` and lets callers
                # that normalise by it produce values above 1.0.
                if limit is not None and len(seen) >= limit:
                    return seen
                queue.append(neighbour)
    return seen


def open_cell_count(blocked, cols: int, rows: int) -> int:
    """Number of cells not occupied by `blocked`."""
    return cols * rows - len(blocked)
