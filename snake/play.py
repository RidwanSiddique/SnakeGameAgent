"""Play or watch a game in a desktop window.

    python3 -m snake.play                  # watch the trained agent
    python3 -m snake.play --level 3        # on a particular board
    python3 -m snake.play --human          # play it yourself, arrow keys
    python3 -m snake.play --checkpoint agent.pth

This is the only entry point that opens a window. Training never imports it, so
training never needs a display.
"""
import argparse

import pygame

from .core.engine import SnakeEngine
from .core.levels import load_levels
from .core.state import FEATURE_VERSION
from .core.types import CLOCKWISE, Action, Direction
from .render.pygame_view import PygameView

KEY_DIRECTIONS = {
    pygame.K_UP: Direction.UP,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_w: Direction.UP,
    pygame.K_s: Direction.DOWN,
    pygame.K_a: Direction.LEFT,
    pygame.K_d: Direction.RIGHT,
}


def action_for(current: Direction, desired: Direction) -> Action:
    """Convert an absolute heading into a turn, ignoring impossible reversals."""
    if desired is current:
        return Action.STRAIGHT
    index = CLOCKWISE.index(current)
    if CLOCKWISE[(index + 1) % 4] is desired:
        return Action.RIGHT
    if CLOCKWISE[(index - 1) % 4] is desired:
        return Action.LEFT
    return Action.STRAIGHT  # a reversal: keep going rather than turn into the neck


def load_policy(checkpoint: str):
    """Return a callable choosing an action, or None if no agent is available."""
    import torch

    from .core.state import get_state
    from .train.model import Linear_QNet

    model, meta = Linear_QNet.load(checkpoint, expect_feature_version=FEATURE_VERSION)
    model.eval()
    print(f"loaded {checkpoint}: {meta.describe()}")

    def policy(engine) -> Action:
        with torch.no_grad():
            q_values = model(torch.as_tensor(get_state(engine), dtype=torch.float))
        return Action(int(torch.argmax(q_values).item()))

    return policy


def main():
    parser = argparse.ArgumentParser(description="Watch the agent play, or play yourself.")
    parser.add_argument("--level", type=int, default=1, help="level id (1-4)")
    parser.add_argument("--human", action="store_true", help="take the controls yourself")
    parser.add_argument("--checkpoint", default="agent_best.pth")
    parser.add_argument("--fps", type=int, default=0, help="frame rate (default: 10 human, 18 agent)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    levels = load_levels()
    if args.level not in levels:
        parser.error(f"level must be one of {sorted(levels)}")

    engine = SnakeEngine(levels[args.level], seed=args.seed)
    policy = None
    if not args.human:
        try:
            policy = load_policy(args.checkpoint)
        except (FileNotFoundError, ValueError) as cause:
            parser.error(f"{cause}\nTrain one first: python3 -m snake.train")

    fps = args.fps or (10 if args.human else 18)
    view = PygameView(engine, fps=fps, caption=f"Snake — {'you' if args.human else 'agent'}")

    heading = engine.direction
    best = 0
    running = True

    while running:
        running = view.pump()

        if args.human:
            pressed = pygame.key.get_pressed()
            for key, direction in KEY_DIRECTIONS.items():
                if pressed[key]:
                    heading = direction
                    break
            action = action_for(engine.direction, heading)
        else:
            action = policy(engine)

        if engine.step(action).died:
            best = max(best, engine.score)
            print(f"score {engine.score}  (best {best})")
            engine.reset(engine.seed + 1)
            heading = engine.direction

        view.render(f"best {best}   ·   esc to quit")

    view.close()


if __name__ == "__main__":
    main()
