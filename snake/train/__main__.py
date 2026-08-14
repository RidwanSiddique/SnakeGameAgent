"""CLI: python3 -m snake.train [--episodes N] [--levels 1 2 3 4]"""
import argparse

from .loop import train


def main():
    parser = argparse.ArgumentParser(description="Train the snake agent across levels.")
    parser.add_argument("--episodes", type=int, default=2000, help="episodes to train")
    parser.add_argument("--seed", type=int, default=1, help="seed for reproducible runs")
    parser.add_argument("--levels", type=int, nargs="+", default=None,
                        help="restrict training to these level ids (default: full curriculum)")
    parser.add_argument("--name", default="agent.pth", help="checkpoint file name")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        seed=args.seed,
        level_ids=args.levels,
        checkpoint_name=args.name,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
