"""Watch a trained agent play: python3 -m snake.play [--level N] [--name FILE]

Rendering is entirely separate from the engine, so this script observes the same
logic training used — it does not re-implement anything.
"""
import argparse

from .core.engine import SnakeEngine
from .core.levels import load_levels
from .core.state import FEATURE_VERSION, get_state
from .train.agent import Agent
from .train.model import Linear_QNet


def main():
    parser = argparse.ArgumentParser(description="Watch the trained agent play.")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--name", default="agent.pth")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    model, meta = Linear_QNet.load(args.name, expect_feature_version=FEATURE_VERSION)
    print(f"loaded {args.name}: {meta.describe()}")

    from .render.pygame_view import PygameView  # imported late: needs a display

    levels = load_levels()
    engine = SnakeEngine(levels[args.level], seed=args.seed)
    agent = Agent(model=model)
    view = PygameView(engine, fps=args.fps)

    episode, best = 1, 0
    while view.pump():
        if engine.step(agent.get_action(get_state(engine), greedy=True)).died:
            best = max(best, engine.score)
            episode += 1
            engine.reset(seed=args.seed + episode)
        view.render(f"episode {episode}   best {best}   esc to quit")

    view.close()


if __name__ == "__main__":
    main()
