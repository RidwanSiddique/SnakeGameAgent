"""Value types shared by the engine, the trainer, and the renderer.

Coordinates are *grid cells*, not pixels. The original game stored pixel
positions (multiples of BLOCK_SIZE) inside the engine, which tied level data to
one canvas size; cells keep `shared/levels.json` resolution-independent and let
the renderer scale freely.
"""
from enum import Enum
from typing import NamedTuple


class Point(NamedTuple):
    """A grid cell. Integer column and row, origin top-left."""

    x: int
    y: int


class Direction(Enum):
    """Facing of the snake's head. Values match the original game.py."""

    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Clockwise order. A right turn steps forward through this list, a left turn
# steps back. The engine and the TypeScript port must agree on this ordering.
CLOCKWISE = (Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP)

DELTA = {
    Direction.RIGHT: (1, 0),
    Direction.LEFT: (-1, 0),
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
}


class Action(Enum):
    """Turn relative to current heading, which is what the network outputs."""

    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2

    @classmethod
    def from_one_hot(cls, one_hot) -> "Action":
        """Accept the `[1, 0, 0]` vectors the existing agent produces."""
        for index, flag in enumerate(one_hot):
            if flag:
                return cls(index)
        raise ValueError(f"no action set in one-hot vector: {one_hot!r}")


class StepResult(NamedTuple):
    """What happened during one engine step.

    Deliberately reports *facts*, not reward. Reward shaping is a training
    concern and lives in the trainer; keeping it out of the engine is what lets
    the same engine run unchanged in the browser, and it removes the cross-episode
    state that made the old `agent.last_distance` leak between games.
    """

    ate: bool
    died: bool
    score: int
    steps: int

    @property
    def done(self) -> bool:
        return self.died
