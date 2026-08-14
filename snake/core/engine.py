"""The snake game as pure logic.

No pygame, no clock, no I/O, no printing. Two consequences follow, and both are
the point of this module:

  - training runs as fast as the CPU allows instead of the 40 steps/second the
    old `clock.tick(SPEED)` imposed
  - the same rules can be reimplemented in TypeScript and checked frame-by-frame
    against recorded trajectories

Determinism is a hard requirement: a given (level, seed, action sequence) must
produce byte-identical play here and in the browser, or races are not fair and
the trained agent is not the deployed agent.
"""
from .levels import Level, resolve_obstacles, spawn_cells
from .rng import Rng
from .types import CLOCKWISE, DELTA, Action, Direction, Point, StepResult

# An episode ends if the snake goes this many steps per body segment without
# eating. Carried over from the original game.py, which used the same rule to
# stop the agent circling forever.
STALL_STEPS_PER_SEGMENT = 100


class SnakeEngine:
    """One playable game. Drive it with `reset()` then repeated `step()`."""

    def __init__(self, level: Level, seed: int = 0):
        self.level = level
        self.grid = level.grid

        # A fixed level's geometry cannot change between episodes, so expand and
        # validate it once here rather than on every reset. Validation walks the
        # whole grid, and training resets millions of times. This is safe for
        # determinism because fixed levels draw nothing from the RNG.
        self._fixed_obstacles = None if level.is_procedural else resolve_obstacles(level, Rng(0))

        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        """Start a new episode. Reusing the previous seed replays it exactly."""
        if seed is not None:
            self._seed = seed
        self._rng = Rng(self._seed)

        # Procedural layouts are redrawn per episode and consume the generator;
        # fixed layouts were resolved once in __init__ and consume nothing.
        if self._fixed_obstacles is None:
            self.obstacles = resolve_obstacles(self.level, self._rng)
        else:
            self.obstacles = self._fixed_obstacles

        self.snake = list(spawn_cells(self.grid))
        # Mirrors `snake` as a set. The engine is collision-query bound — the
        # state encoder's distance rays alone probe ~100 cells per frame — and a
        # list scan there made encoding the dominant cost of training.
        self._body = set(self.snake)
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self._steps_since_food = 0
        self.food = None
        self._place_food()

    @property
    def head(self) -> Point:
        return self.snake[0]

    @property
    def seed(self) -> int:
        return self._seed

    def is_collision(self, cell: Point | None = None, *, ignore_tail: bool = False) -> bool:
        """True if `cell` is a wall, an obstacle, or the snake's own body.

        `ignore_tail` excludes the final segment, which vacates the cell on the
        next step. The state encoder uses it so the agent is not scared away from
        a square that will be free by the time it arrives.
        """
        cell = self.head if cell is None else cell

        if not self.grid.contains(cell):
            return True
        if cell in self.obstacles:
            return True
        if cell not in self._body:
            return False

        # In the body: the head never blocks (it is where we are), and the tail
        # only blocks when it is not about to move out of the way.
        if cell == self.snake[0]:
            return False
        return not (ignore_tail and cell == self.snake[-1])

    def step(self, action) -> StepResult:
        """Advance one tick and report what happened."""
        if not isinstance(action, Action):
            action = Action.from_one_hot(action)

        self.steps += 1
        self._steps_since_food += 1

        self.direction = self._turn(self.direction, action)
        dx, dy = DELTA[self.direction]
        new_head = Point(self.head.x + dx, self.head.y + dy)

        # Check before moving: the tail only vacates its cell if the snake does
        # not grow this step, and it does not grow unless it eats.
        if self.is_collision(new_head, ignore_tail=new_head != self.food):
            return StepResult(ate=False, died=True, score=self.score, steps=self.steps)

        if self._steps_since_food > STALL_STEPS_PER_SEGMENT * len(self.snake):
            return StepResult(ate=False, died=True, score=self.score, steps=self.steps)

        self.snake.insert(0, new_head)
        self._body.add(new_head)

        ate = new_head == self.food
        if ate:
            self.score += 1
            self._steps_since_food = 0
            self._place_food()
        else:
            self._body.discard(self.snake.pop())

        return StepResult(ate=ate, died=False, score=self.score, steps=self.steps)

    @staticmethod
    def _turn(direction: Direction, action: Action) -> Direction:
        index = CLOCKWISE.index(direction)
        if action is Action.RIGHT:
            index = (index + 1) % 4
        elif action is Action.LEFT:
            index = (index - 1) % 4
        return CLOCKWISE[index]

    def _place_food(self) -> None:
        """Put food on a uniformly chosen free cell.

        Rejection sampling would be simpler, but its number of RNG draws depends
        on how full the board is, and the TypeScript port would have to match
        that draw count exactly. Enumerating free cells and making a single draw
        keeps the two implementations trivially in sync.
        """
        occupied = self._body | self.obstacles
        free = [
            Point(x, y)
            for y in range(self.grid.rows)
            for x in range(self.grid.cols)
            if Point(x, y) not in occupied
        ]

        if not free:
            self.food = None  # board solved; nowhere left to place food
            return

        self.food = free[self._rng.randint(0, len(free) - 1)]

    def snapshot(self) -> dict:
        """Serialisable state, used by the golden-trajectory recorder."""
        return {
            "head": list(self.head),
            "snake": [list(c) for c in self.snake],
            "food": list(self.food) if self.food else None,
            "direction": self.direction.name,
            "score": self.score,
            "steps": self.steps,
        }
