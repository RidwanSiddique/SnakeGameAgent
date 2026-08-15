"""Loading and resolving level definitions from shared/levels.json.

Levels are data, not code. The same schema describes built-in levels and boards
drawn by visitors in the web designer, so the designer mode needs no separate
format and no separate validation path.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from .grid import flood_fill, open_cell_count
from .types import Point

LEVELS_PATH = Path(__file__).resolve().parents[2] / "shared" / "levels.json"

# Cells that must stay clear so the snake never spawns into a wall or a
# dead-end. Relative to the head, in the direction of travel (+x).
SPAWN_CLEARANCE_AHEAD = 5
SPAWN_CLEARANCE_BEHIND = 3


class LevelError(ValueError):
    """Raised for a malformed or unplayable level definition."""


@dataclass(frozen=True)
class Grid:
    cols: int
    rows: int

    def contains(self, cell: Point) -> bool:
        return 0 <= cell.x < self.cols and 0 <= cell.y < self.rows


@dataclass(frozen=True)
class Level:
    id: int
    name: str
    difficulty: int
    grid: Grid
    obstacles: dict

    @property
    def is_procedural(self) -> bool:
        return self.obstacles.get("kind") == "procedural"


def _expand_fixed(spec: dict, grid: Grid) -> set[Point]:
    """Expand `blocks` rectangles and `cells` pairs into a set of cells."""
    cells: set[Point] = set()

    for block in spec.get("blocks", []):
        for dx in range(block["w"]):
            for dy in range(block["h"]):
                cells.add(Point(block["x"] + dx, block["y"] + dy))

    for pair in spec.get("cells", []):
        cells.add(Point(pair[0], pair[1]))

    out_of_bounds = [c for c in cells if not grid.contains(c)]
    if out_of_bounds:
        raise LevelError(f"obstacles outside the {grid.cols}x{grid.rows} grid: {out_of_bounds[:5]}")

    return cells


def spawn_cells(grid: Grid) -> list[Point]:
    """The snake's starting body, head first, facing right.

    Defined here rather than in the engine because the loader must validate that
    a level leaves this space free.
    """
    head = Point(grid.cols // 2, grid.rows // 2)
    return [head, Point(head.x - 1, head.y), Point(head.x - 2, head.y)]


def spawn_corridor(grid: Grid) -> set[Point]:
    """Cells that must be obstacle-free for the opening moves to be survivable."""
    head = spawn_cells(grid)[0]
    corridor = set()
    for offset in range(-SPAWN_CLEARANCE_BEHIND, SPAWN_CLEARANCE_AHEAD + 1):
        cell = Point(head.x + offset, head.y)
        if grid.contains(cell):
            corridor.add(cell)
    return corridor


def _generate_procedural(spec: dict, grid: Grid, rng) -> set[Point]:
    """Build a random obstacle layout that is guaranteed playable.

    Two properties are enforced, because a randomly generated board can easily be
    neither: total coverage stays under `coverage_limit`, and every open cell
    stays reachable from spawn. Without the reachability check the trainer would
    occasionally hand the agent a board with the food sealed behind a wall and
    punish it for failing an impossible task.
    """
    min_blocks, max_blocks = spec.get("block_count", [6, 12])
    min_size, max_size = spec.get("block_size", [1, 3])
    coverage_limit = spec.get("coverage_limit", 0.15)

    protected = spawn_corridor(grid)
    max_cells = int(grid.cols * grid.rows * coverage_limit)

    for _ in range(30):  # retry whole layouts until one is playable
        cells: set[Point] = set()
        for _ in range(rng.randint(min_blocks, max_blocks)):
            w = rng.randint(min_size, max_size)
            h = rng.randint(min_size, max_size)
            x = rng.randint(0, max(0, grid.cols - w))
            y = rng.randint(0, max(0, grid.rows - h))
            candidate = {Point(x + dx, y + dy) for dx in range(w) for dy in range(h)}
            if candidate & protected:
                continue
            if len(cells | candidate) > max_cells:
                continue
            cells |= candidate

        if _is_fully_reachable(cells, grid):
            return cells

    # Every attempt failed, which means the parameters are too aggressive for
    # this grid. An open board is a poor level but a correct one; failing here
    # would abort a training run for a recoverable reason.
    return set()


def _is_fully_reachable(obstacles: set[Point], grid: Grid) -> bool:
    """True if no open cell is walled off from the spawn point."""
    head = spawn_cells(grid)[0]
    reached = flood_fill(head, obstacles, grid.cols, grid.rows)
    return len(reached) == open_cell_count(obstacles, grid.cols, grid.rows)


def resolve_obstacles(level: Level, rng) -> frozenset[Point]:
    """Produce the concrete obstacle cells for one episode.

    Fixed levels ignore `rng` and return the same cells every time. Procedural
    levels consume it, so the layout is reproducible from the episode seed —
    which is what lets a race replay a board exactly.
    """
    if level.is_procedural:
        return frozenset(_generate_procedural(level.obstacles, level.grid, rng))

    cells = _expand_fixed(level.obstacles, level.grid)
    conflict = cells & spawn_corridor(level.grid)
    if conflict:
        raise LevelError(
            f"level {level.id} ({level.name}) puts obstacles in the spawn corridor: {sorted(conflict)}"
        )
    if not _is_fully_reachable(cells, level.grid):
        raise LevelError(f"level {level.id} ({level.name}) walls off part of the board")
    return frozenset(cells)


def load_levels(path: Path | None = None) -> dict[int, Level]:
    """Read and validate every level definition."""
    path = path or LEVELS_PATH
    data = json.loads(path.read_text())

    if data.get("schema_version") != 1:
        raise LevelError(f"unsupported schema_version: {data.get('schema_version')!r}")

    grid = Grid(cols=data["grid"]["cols"], rows=data["grid"]["rows"])

    levels: dict[int, Level] = {}
    for entry in data["levels"]:
        level = Level(
            id=entry["id"],
            name=entry["name"],
            difficulty=entry.get("difficulty", 1),
            grid=grid,
            obstacles=entry["obstacles"],
        )
        if level.id in levels:
            raise LevelError(f"duplicate level id: {level.id}")
        levels[level.id] = level
    return levels


def level_from_cells(cells, grid: Grid, level_id: int = 0, name: str = "Custom") -> Level:
    """Build a Level from a designer-drawn cell list, without touching disk."""
    return Level(
        id=level_id,
        name=name,
        difficulty=0,
        grid=grid,
        obstacles={"kind": "fixed", "blocks": [], "cells": [[c.x, c.y] for c in cells]},
    )
