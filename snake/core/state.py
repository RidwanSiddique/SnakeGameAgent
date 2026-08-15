"""Feature encoding: turns an engine into the vector the network consumes.

The original 14 features are preserved in their original order and meaning, so
previously trained behaviour remains comparable and the diff stays legible. Seven
features are appended, all aimed at one failure: the old agent could not perceive
enclosed space, so it routinely sealed itself into a pocket it had no way to see
coming. Hand-drawn levels from the web designer are pocket factories, which makes
this the difference between an agent that looks clever and one that looks broken.

  index   feature
  0-2     immediate danger straight / right / left        (original)
  3-5     path blocked within 3 steps s / r / l           (original)
  6-9     current direction, one-hot                      (original)
  10-13   food direction, one-hot-ish                     (original)
  14-16   free space straight / right / left              (new)
  17      tail reachable after moving straight            (new)
  18-20   normalised distance to obstruction s / r / l    (new)
  21-23   shortest path to food turns s / r / l           (new)
  24      food reachable at all                           (new)

Features 10-13 give only a *bearing* to the food, which is enough on an open
board and useless in a maze: the bearing points through the wall. Measured on
the Corridors level, an agent with bearing alone never ate at all — episodes
ended at the stall limit with score 0. Features 21-24 add the first step of an
actual shortest path, which is what makes obstacle-dense boards — including the
ones visitors draw in the web designer — playable.

FEATURE_VERSION is recorded in every checkpoint. A model trained under one
version cannot be loaded under another, because the input layer would silently
mean something different.
"""
from collections import deque

import numpy as np

from .grid import flood_fill, neighbours
from .types import CLOCKWISE, DELTA, Direction, Point

FEATURE_VERSION = 3
FEATURE_COUNT = 25

LOOKAHEAD_STEPS = 3

PATH_SEARCH_BUDGET = None  # bounded by the board; see _path_direction


def _relative_directions(direction: Direction) -> tuple[Direction, Direction, Direction]:
    """Return (straight, right, left) as absolute directions."""
    index = CLOCKWISE.index(direction)
    return (
        direction,
        CLOCKWISE[(index + 1) % 4],
        CLOCKWISE[(index - 1) % 4],
    )


def _step(cell: Point, direction: Direction, times: int = 1) -> Point:
    dx, dy = DELTA[direction]
    return Point(cell.x + dx * times, cell.y + dy * times)


def _path_blocked(engine, direction: Direction, steps: int = LOOKAHEAD_STEPS) -> bool:
    """True if anything obstructs the next `steps` cells in `direction`."""
    for distance in range(1, steps + 1):
        if engine.is_collision(_step(engine.head, direction, distance)):
            return True
    return False


def _distance_to_obstruction(engine, direction: Direction, cap: int) -> float:
    """Normalised free run in `direction`: 0.0 means blocked immediately."""
    for distance in range(1, cap + 1):
        if engine.is_collision(_step(engine.head, direction, distance)):
            return (distance - 1) / cap
    return 1.0


def _free_space(engine, direction: Direction, blocked, budget: int) -> float:
    """How much open room lies through `direction`, relative to body length.

    Capped at the snake's own length: the agent does not need an exact area, only
    whether the space is large enough to survive in. 1.0 means "at least as much
    room as I am long" — the standard safety test — and anything below that is a
    developing trap. The cap also keeps this affordable at training speed.
    """
    reachable = flood_fill(_step(engine.head, direction), blocked, engine.grid.cols, engine.grid.rows, limit=budget)
    return len(reachable) / budget


def _tail_reachable(engine, direction: Direction, blocked) -> bool:
    """Can the snake still reach its own tail after stepping `direction`?

    If it can, a survivable route almost always exists: following the tail is
    safe because the tail keeps retreating. This is the single strongest
    anti-self-trapping signal available at this cost.
    """
    if len(engine.snake) < 2:
        return True
    tail = engine.snake[-1]
    start = _step(engine.head, direction)
    if start == tail:
        return True

    # Cap the search at twice the body length. An uncapped fill scans the whole
    # board every frame and dominated training cost. The cap loses nothing that
    # matters: this feature detects *traps*, and a region larger than the snake
    # is not a trap — following the tail out of it is always available.
    budget = 2 * len(engine.snake) + 8
    reached = flood_fill(start, blocked - {tail}, engine.grid.cols, engine.grid.rows, limit=budget)
    return tail in reached or len(reached) >= budget


def _compute_path(engine, blocked) -> tuple[Direction | None, int | None]:
    """Shortest route from head to food: (first step, length).

    Breadth-first from the head, tagging every reached cell with the *initial*
    move that leads to it, so the answer falls out when the food is reached
    without reconstructing a parent chain. Returns (None, None) when no route
    exists, which is itself worth telling the agent about.
    """
    food = engine.food
    if food is None:
        return None, None

    cols, rows = engine.grid.cols, engine.grid.rows
    head = engine.head

    origin: dict[Point, Direction] = {}
    distance: dict[Point, int] = {}
    queue: deque[Point] = deque()

    for direction in CLOCKWISE:
        cell = _step(head, direction)
        if 0 <= cell.x < cols and 0 <= cell.y < rows and cell not in blocked:
            if cell == food:
                return direction, 1
            origin[cell] = direction
            distance[cell] = 1
            queue.append(cell)

    # No artificial visit cap. A BFS frontier expands as a diamond, so capping it
    # at a few hundred cells silently truncates the search well before the far
    # side of the board and reports "unreachable" for food in plain sight. The
    # search is already bounded by the cell count, which is the real limit.
    while queue:
        cell = queue.popleft()
        for neighbour in neighbours(cell, cols, rows):
            if neighbour in origin or neighbour in blocked or neighbour == head:
                continue
            if neighbour == food:
                return origin[cell], distance[cell] + 1
            origin[neighbour] = origin[cell]
            distance[neighbour] = distance[cell] + 1
            queue.append(neighbour)

    return None, None


def path_to_food(engine) -> tuple[Direction | None, int | None]:
    """Cached `_compute_path` for the engine's current position.

    Both the encoder and the trainer's reward shaping need this, and they run
    against the same position on every step. Computing it twice would double the
    dominant cost of training, so the result is memoised against a key that
    changes whenever the board does.
    """
    key = (engine.steps, engine.head, engine.food, len(engine.snake))
    cached = getattr(engine, "_path_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    blocked = set(engine.obstacles) | set(engine.snake[:-1])
    result = _compute_path(engine, blocked)
    engine._path_cache = (key, result)
    return result


def get_state(engine) -> np.ndarray:
    """Encode the engine's current position as a float32 feature vector."""
    head = engine.head
    straight, right, left = _relative_directions(engine.direction)

    # The tail vacates its cell as the snake advances, so treating it as solid
    # would make the agent flinch from squares that are about to be free.
    blocked = set(engine.obstacles) | set(engine.snake[:-1])

    budget = max(4, len(engine.snake))
    ray_cap = max(engine.grid.cols, engine.grid.rows)

    food = engine.food or head
    path_direction, _ = path_to_food(engine)

    features = [
        # 0-2: immediate danger
        engine.is_collision(_step(head, straight)),
        engine.is_collision(_step(head, right)),
        engine.is_collision(_step(head, left)),

        # 3-5: obstruction within the next few cells
        _path_blocked(engine, straight),
        _path_blocked(engine, right),
        _path_blocked(engine, left),

        # 6-9: current heading
        engine.direction is Direction.LEFT,
        engine.direction is Direction.RIGHT,
        engine.direction is Direction.UP,
        engine.direction is Direction.DOWN,

        # 10-13: where the food is
        food.x < head.x,
        food.x > head.x,
        food.y < head.y,
        food.y > head.y,

        # 14-16: room to manoeuvre
        _free_space(engine, straight, blocked, budget),
        _free_space(engine, right, blocked, budget),
        _free_space(engine, left, blocked, budget),

        # 17: escape route still open
        _tail_reachable(engine, straight, blocked),

        # 18-20: how far before something stops us
        _distance_to_obstruction(engine, straight, ray_cap),
        _distance_to_obstruction(engine, right, ray_cap),
        _distance_to_obstruction(engine, left, ray_cap),

        # 21-24: which way an actual route to the food starts, and whether one
        # exists at all. Unlike the bearing at 10-13, this respects walls.
        path_direction is straight,
        path_direction is right,
        path_direction is left,
        path_direction is not None,
    ]

    return np.array(features, dtype=np.float32)
