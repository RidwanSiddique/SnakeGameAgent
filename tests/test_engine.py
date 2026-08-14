"""Tests for the headless engine and level loader."""
import pytest

from snake.core.engine import STALL_STEPS_PER_SEGMENT, SnakeEngine
from snake.core.grid import flood_fill
from snake.core.levels import (
    Grid,
    LevelError,
    level_from_cells,
    load_levels,
    resolve_obstacles,
    spawn_corridor,
)
from snake.core.rng import Rng
from snake.core.types import Action, Direction, Point


@pytest.fixture(scope="module")
def levels():
    return load_levels()


@pytest.fixture
def open_level(levels):
    return levels[1]


# --- level loading -----------------------------------------------------------


def test_flood_fill_respects_its_limit():
    """A per-pop check overshoots, since one pop can enqueue four cells.

    Callers normalise by `limit`, so overshoot produced feature values above 1.0.
    """
    for limit in range(1, 12):
        reached = flood_fill(Point(16, 12), set(), 32, 24, limit=limit)
        assert len(reached) <= limit


def test_flood_fill_without_a_limit_covers_the_open_board():
    assert len(flood_fill(Point(16, 12), set(), 32, 24)) == 32 * 24


def test_flood_fill_stops_at_walls():
    walls = {Point(3, y) for y in range(24)}
    reached = flood_fill(Point(0, 0), walls, 32, 24)
    assert len(reached) == 3 * 24
    assert Point(4, 0) not in reached


def test_all_levels_load(levels):
    assert set(levels) == {1, 2, 3, 4}
    assert levels[2].name == "Scattered Blocks"


def test_every_level_resolves_to_a_playable_board(levels):
    """Catches obstacles in the spawn corridor and walled-off regions."""
    for level in levels.values():
        obstacles = resolve_obstacles(level, Rng(1))
        assert not obstacles & spawn_corridor(level.grid)


def test_fixed_levels_are_stable_across_seeds(levels):
    a = resolve_obstacles(levels[2], Rng(1))
    b = resolve_obstacles(levels[2], Rng(9999))
    assert a == b


def test_procedural_level_varies_with_seed(levels):
    a = resolve_obstacles(levels[4], Rng(1))
    b = resolve_obstacles(levels[4], Rng(2))
    assert a != b


def test_procedural_level_is_reproducible(levels):
    assert resolve_obstacles(levels[4], Rng(7)) == resolve_obstacles(levels[4], Rng(7))


def test_procedural_boards_never_wall_off_a_region(levels):
    """The trainer must never be handed a board with unreachable food."""
    grid = levels[4].grid
    for seed in range(40):
        obstacles = resolve_obstacles(levels[4], Rng(seed))
        reachable = flood_fill(Point(grid.cols // 2, grid.rows // 2), obstacles, grid.cols, grid.rows)
        assert len(reachable) == grid.cols * grid.rows - len(obstacles)


def test_obstacles_in_spawn_corridor_are_rejected():
    grid = Grid(cols=32, rows=24)
    bad = level_from_cells([Point(grid.cols // 2 + 1, grid.rows // 2)], grid)
    with pytest.raises(LevelError, match="spawn corridor"):
        resolve_obstacles(bad, Rng(0))


def test_walled_off_region_is_rejected():
    """A designer-drawn board that seals off a corner must not be accepted."""
    grid = Grid(cols=32, rows=24)
    wall = [Point(3, y) for y in range(grid.rows)] + [Point(x, 3) for x in range(4)]
    with pytest.raises(LevelError, match="walls off"):
        resolve_obstacles(level_from_cells(wall, grid), Rng(0))


def test_out_of_bounds_obstacle_is_rejected():
    grid = Grid(cols=32, rows=24)
    with pytest.raises(LevelError, match="outside"):
        resolve_obstacles(level_from_cells([Point(99, 99)], grid), Rng(0))


# --- engine ------------------------------------------------------------------


def test_starts_facing_right_with_three_segments(open_level):
    engine = SnakeEngine(open_level, seed=1)
    assert len(engine.snake) == 3
    assert engine.direction is Direction.RIGHT
    assert engine.score == 0


def test_same_seed_and_actions_replay_identically(open_level):
    actions = [Action(i % 3) for i in range(200)]

    def run():
        engine = SnakeEngine(open_level, seed=42)
        trace = []
        for action in actions:
            result = engine.step(action)
            trace.append(engine.snapshot())
            if result.died:
                break
        return trace

    assert run() == run()


def test_different_seeds_place_food_differently(open_level):
    a = SnakeEngine(open_level, seed=1)
    b = SnakeEngine(open_level, seed=2)
    assert a.food != b.food


def test_straight_moves_one_cell_right(open_level):
    engine = SnakeEngine(open_level, seed=1)
    before = engine.head
    engine.step(Action.STRAIGHT)
    assert engine.head == Point(before.x + 1, before.y)


def test_right_turn_from_right_faces_down(open_level):
    engine = SnakeEngine(open_level, seed=1)
    engine.step(Action.RIGHT)
    assert engine.direction is Direction.DOWN


def test_left_turn_from_right_faces_up(open_level):
    engine = SnakeEngine(open_level, seed=1)
    engine.step(Action.LEFT)
    assert engine.direction is Direction.UP


def test_accepts_one_hot_actions_from_the_existing_agent(open_level):
    engine = SnakeEngine(open_level, seed=1)
    engine.step([0, 1, 0])
    assert engine.direction is Direction.DOWN


def test_running_into_the_wall_ends_the_episode(open_level):
    engine = SnakeEngine(open_level, seed=1)
    result = None
    for _ in range(open_level.grid.cols + 5):
        result = engine.step(Action.STRAIGHT)
        if result.died:
            break
    assert result.died


def test_body_length_grows_only_when_eating(open_level):
    engine = SnakeEngine(open_level, seed=1)
    length = len(engine.snake)
    for _ in range(5):
        result = engine.step(Action.STRAIGHT)
        if result.ate:
            assert len(engine.snake) == length + 1
        else:
            assert len(engine.snake) == length
        length = len(engine.snake)


def test_food_never_spawns_on_the_snake_or_an_obstacle(levels):
    for level_id in (1, 2, 3, 4):
        engine = SnakeEngine(levels[level_id], seed=level_id)
        for _ in range(300):
            assert engine.food not in engine.snake
            assert engine.food not in engine.obstacles
            if engine.step(Action.STRAIGHT).died:
                engine.reset()


def test_stalling_ends_the_episode(open_level):
    """Circling forever must terminate, or training episodes never end."""
    engine = SnakeEngine(open_level, seed=1)
    limit = STALL_STEPS_PER_SEGMENT * len(engine.snake) + 50
    died = False
    for i in range(limit):
        # A repeating right-turn loop revisits the same few cells without eating.
        if engine.step(Action.RIGHT if i % 2 == 0 else Action.STRAIGHT).died:
            died = True
            break
    assert died


def test_obstacles_are_solid(levels):
    engine = SnakeEngine(levels[2], seed=1)
    obstacle = next(iter(engine.obstacles))
    assert engine.is_collision(obstacle)


def test_reset_restores_the_opening_position(open_level):
    engine = SnakeEngine(open_level, seed=3)
    opening = engine.snapshot()
    for _ in range(10):
        if engine.step(Action.STRAIGHT).died:
            break
    engine.reset()
    assert engine.snapshot() == opening
