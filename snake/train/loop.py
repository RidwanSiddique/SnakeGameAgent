"""The training loop.

Headless by default. The engine imposes no frame rate, so this runs as fast as
the CPU allows rather than the 40 steps/second the original was pinned to.
"""
import time
from collections import deque

from ..core.engine import SnakeEngine
from ..core.levels import load_levels
from ..core.rng import Rng
from ..core.state import FEATURE_VERSION
from .agent import Agent
from .curriculum import Curriculum
from .evaluate import evaluate, summarise

EVAL_EVERY = 250          # episodes
CHECKPOINT_EVERY = 100    # episodes
RECENT_WINDOW = 50        # episodes averaged for the running score


def train(
    *,
    episodes: int = 2000,
    seed: int = 1,
    checkpoint_name: str = "agent.pth",
    level_ids=None,
    log_every: int = 25,
    quiet: bool = False,
):
    """Train one network across the curriculum.

    `level_ids` restricts training to specific levels; the default trains the
    full mix, which is what the web level designer requires.
    """
    levels = load_levels()
    if level_ids:
        levels = {i: levels[i] for i in level_ids}

    agent = Agent()
    curriculum = Curriculum(levels)
    rng = Rng(seed)

    best_score = 0
    recent = deque(maxlen=RECENT_WINDOW)
    seen_levels = set()
    started = time.perf_counter()

    for episode in range(1, episodes + 1):
        level_id = curriculum.sample(agent.n_games, rng)
        seen_levels.add(level_id)

        engine = SnakeEngine(levels[level_id], seed=rng.next_u32())
        agent.start_episode(engine)
        state = agent.get_state(engine)

        while True:
            action = agent.get_action(state)
            result = engine.step(action)
            reward = agent.reward_for(engine, result)
            next_state = agent.get_state(engine)

            agent.train_short_memory(state, action, reward, next_state, result.died)
            agent.remember(state, action, reward, next_state, result.died)
            state = next_state

            if result.died:
                break

        agent.n_games += 1
        agent.train_long_memory()
        recent.append(engine.score)
        best_score = max(best_score, engine.score)

        if not quiet and episode % log_every == 0:
            rate = episode / (time.perf_counter() - started)
            print(
                f"ep {episode:5d}  L{level_id}  "
                f"score {engine.score:3d}  best {best_score:3d}  "
                f"recent {sum(recent)/len(recent):5.2f}  "
                f"eps {agent.epsilon:.2f}  "
                f"{rate:.1f} ep/s  [{curriculum.describe(agent.n_games)}]"
            )

        if episode % CHECKPOINT_EVERY == 0:
            agent.model.save(
                agent.checkpoint(seen_levels, best_score, sum(recent) / len(recent)),
                checkpoint_name,
            )

        if not quiet and episode % EVAL_EVERY == 0:
            print(f"\n-- evaluation at episode {episode} (greedy, fixed seeds) --")
            print(summarise(evaluate(agent, levels, episodes=10)))
            print()

    meta = agent.checkpoint(seen_levels, best_score, sum(recent) / len(recent) if recent else 0.0)
    path = agent.model.save(meta, checkpoint_name)

    if not quiet:
        elapsed = time.perf_counter() - started
        print(f"\ntrained {episodes} episodes in {elapsed:.1f}s ({episodes/elapsed:.1f} ep/s)")
        print(f"saved {path.name}: {meta.describe()}")
        print(f"\n-- final evaluation --")
        print(summarise(evaluate(agent, levels)))

    return agent, meta


__all__ = ["train", "FEATURE_VERSION"]
