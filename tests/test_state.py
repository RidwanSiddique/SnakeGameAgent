"""Tests for the feature encoder.

The free-space and tail-reachability features are checked against hand-built
boards with known answers, because a subtly wrong flood fill would still train
to *something* and the failure would only show up as an agent that plays badly.
"""
import numpy as np
import pytest

from snake.core.engine import SnakeEngine
from snake.core.levels import Grid, level_from_cells, load_levels
from snake.core.state import FEATURE_COUNT, get_state
from snake.core.types import Action, Direction, Point


@pytest.fixture(scope="module")
def levels():
    return load_levels()


def test_vector_has_the_declared_shape(levels):
    state = get_state(SnakeEngine(levels[1], seed=1))
    assert state.shape == (FEATURE_COUNT,)
    assert state.dtype == np.float32


def test_all_features_stay_normalised(levels):
    """Anything outside [0, 1] would dominate the first layer."""
    for level_id in (1, 2, 3, 4):
        engine = SnakeEngine(levels[level_id], seed=level_id)
        for _ in range(400):
            state = get_state(engine)
            assert state.min() >= 0.0
            assert state.max() <= 1.0
            if engine.step(Action.STRAIGHT).died:
                engine.reset()


def test_original_fourteen_features_keep_their_meaning(levels):
    engine = SnakeEngine(levels[1], seed=1)
    state = get_state(engine)
    # Opening position: facing right, mid-board, nothing adjacent.
    assert state[0] == 0.0  # no danger straight
    assert state[7] == 1.0  # direction RIGHT
    assert state[6] == state[8] == state[9] == 0.0


def test_danger_is_flagged_against_a_wall(levels):
    engine = SnakeEngine(levels[1], seed=1)
    while engine.head.x < engine.grid.cols - 1:
        assert not engine.step(Action.STRAIGHT).died
    assert get_state(engine)[0] == 1.0


def test_food_direction_bits(levels):
    engine = SnakeEngine(levels[1], seed=1)
    state = get_state(engine)
    assert state[10] == float(engine.food.x < engine.head.x)
    assert state[11] == float(engine.food.x > engine.head.x)


def _engine_with_obstacles(cells, grid=None):
    grid = grid or Grid(cols=32, rows=24)
    return SnakeEngine(level_from_cells(cells, grid), seed=1)


def test_free_space_is_full_on_an_open_board(levels):
    engine = SnakeEngine(levels[1], seed=1)
    state = get_state(engine)
    assert state[14] == 1.0  # straight
    assert state[15] == 1.0  # right
    assert state[16] == 1.0  # left


def test_free_space_collapses_inside_a_pocket():
    """A three-cell dead end to the agent's right must read as cramped."""
    grid = Grid(cols=32, rows=24)
    head = Point(grid.cols // 2, grid.rows // 2)
    # Box in the cells below the head, leaving a pocket of exactly 2 cells.
    walls = [
        Point(head.x - 1, head.y + 1), Point(head.x + 1, head.y + 1),
        Point(head.x - 1, head.y + 2), Point(head.x + 1, head.y + 2),
        Point(head.x, head.y + 3),
    ]
    engine = _engine_with_obstacles(walls, grid)
    state = get_state(engine)
    assert state[15] < 1.0          # right (downward) is a pocket
    assert state[14] == 1.0         # straight is still open


def test_tail_reachable_is_true_on_an_open_board(levels):
    engine = SnakeEngine(levels[1], seed=1)
    assert get_state(engine)[17] == 1.0


def _drive_to(engine, target_x):
    """Walk the snake straight to a column, keeping the test's setup honest."""
    while engine.head.x < target_x:
        assert not engine.step(Action.STRAIGHT).died


def test_tail_unreachable_when_sealed_in():
    """Head walled into a cul-de-sac: the tail must read as unreachable.

    The cul-de-sac is built outside the spawn corridor, which the level loader
    keeps clear, so the snake is driven into position first.
    """
    grid = Grid(cols=32, rows=24)
    spawn = Point(grid.cols // 2, grid.rows // 2)
    pocket = Point(spawn.x + 8, spawn.y)
    walls = [
        Point(pocket.x + 1, pocket.y),
        Point(pocket.x, pocket.y - 1),
        Point(pocket.x, pocket.y + 1),
    ]
    engine = _engine_with_obstacles(walls, grid)
    _drive_to(engine, pocket.x - 1)
    # The only exit from `pocket` is back through the snake's own body.
    assert get_state(engine)[17] == 0.0


def test_distance_ray_is_zero_when_blocked_immediately():
    grid = Grid(cols=32, rows=24)
    spawn = Point(grid.cols // 2, grid.rows // 2)
    wall = Point(spawn.x + 8, spawn.y)
    engine = _engine_with_obstacles([wall], grid)
    _drive_to(engine, wall.x - 1)
    assert get_state(engine)[18] == 0.0


def test_distance_ray_grows_with_clearance(levels):
    engine = SnakeEngine(levels[1], seed=1)
    near_wall = get_state(engine)[18]
    while engine.head.x < engine.grid.cols - 3:
        engine.step(Action.STRAIGHT)
    assert get_state(engine)[18] < near_wall


def test_encoding_is_deterministic(levels):
    a = SnakeEngine(levels[4], seed=5)
    b = SnakeEngine(levels[4], seed=5)
    for _ in range(50):
        assert np.array_equal(get_state(a), get_state(b))
        a.step(Action.STRAIGHT)
        b.step(Action.STRAIGHT)
