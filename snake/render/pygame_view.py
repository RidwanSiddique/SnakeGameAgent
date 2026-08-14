"""Pygame rendering for the headless engine.

The only module in the project that imports pygame. Training never touches it,
which is what lets training run without a display and without a frame cap. A view
observes an engine; it never advances one.
"""
from pathlib import Path

import pygame

from ..core.types import Point

BACKGROUND = (12, 14, 20)
GRID_LINE = (24, 28, 38)
OBSTACLE = (78, 84, 102)
SNAKE_BODY = (56, 132, 255)
SNAKE_INNER = (120, 180, 255)
SNAKE_HEAD = (140, 200, 255)
FOOD = (236, 72, 88)
TEXT = (232, 236, 245)

FONT_PATH = Path(__file__).resolve().parents[2] / "arial.ttf"


class PygameView:
    """Draws an engine's current position to a window."""

    def __init__(self, engine, cell_size: int = 20, fps: int = 40, caption: str | None = None):
        pygame.init()
        self.engine = engine
        self.cell_size = cell_size
        self.fps = fps

        self.width = engine.grid.cols * cell_size
        self.height = engine.grid.rows * cell_size
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(caption or f"Snake — {engine.level.name}")

        self.clock = pygame.time.Clock()
        self.font = self._load_font(20)
        self.small_font = self._load_font(14)

    @staticmethod
    def _load_font(size: int):
        """Prefer the bundled font, but never crash if it has moved."""
        if FONT_PATH.exists():
            return pygame.font.Font(str(FONT_PATH), size)
        return pygame.font.SysFont(None, size + 4)

    def _rect(self, cell: Point, inset: int = 0) -> pygame.Rect:
        return pygame.Rect(
            cell.x * self.cell_size + inset,
            cell.y * self.cell_size + inset,
            self.cell_size - inset * 2,
            self.cell_size - inset * 2,
        )

    def pump(self) -> bool:
        """Process window events. Returns False when the user closes the window.

        Callers decide what to do about that. The old game called `quit()` from
        inside its step function, which made the environment impossible to embed.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def render(self, status: str = "") -> None:
        engine = self.engine
        self.display.fill(BACKGROUND)

        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.display, GRID_LINE, (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.display, GRID_LINE, (0, y), (self.width, y))

        for cell in engine.obstacles:
            pygame.draw.rect(self.display, OBSTACLE, self._rect(cell))

        for index, cell in enumerate(engine.snake):
            colour = SNAKE_HEAD if index == 0 else SNAKE_BODY
            pygame.draw.rect(self.display, colour, self._rect(cell))
            if index > 0:
                pygame.draw.rect(self.display, SNAKE_INNER, self._rect(cell, inset=5))

        if engine.food:
            pygame.draw.rect(self.display, FOOD, self._rect(engine.food, inset=3))

        heading = self.font.render(
            f"Score {engine.score}   ·   {engine.level.name}", True, TEXT
        )
        self.display.blit(heading, (8, 6))
        if status:
            self.display.blit(self.small_font.render(status, True, TEXT), (8, 30))

        pygame.display.flip()
        if self.fps:
            self.clock.tick(self.fps)

    def close(self) -> None:
        pygame.quit()
