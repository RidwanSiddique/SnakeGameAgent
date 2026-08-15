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

    best_name = checkpoint_name.replace(".pth", "_best.pth")
    best_eval, best_episode = float("-inf"), 0
    eval_history: list[tuple[int, float]] = []

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

        if episode % EVAL_EVERY == 0:
            reports = evaluate(agent, levels, episodes=10)
            overall = sum(report.mean for report in reports) / len(reports)
            eval_history.append((episode, overall))

            # Keep the best-evaluating weights, not the most recent ones. Q-learning
            # can collapse late in a run — an earlier version of this loop saved only
            # on a schedule and shipped a model scoring 34 after the same run had
            # already reached 83 — so the peak has to be captured when it happens.
            if overall > best_eval:
                best_eval, best_episode = overall, episode
                agent.model.save(agent.checkpoint(seen_levels, best_score, overall), best_name)
                marker = f"  <- new best, saved to {best_name}"
            else:
                marker = f"  (best {best_eval:.2f} @ ep {best_episode})"

            if not quiet:
                print(f"\n-- evaluation at episode {episode} (greedy, fixed seeds) --")
                print(summarise(reports))
                print(f"overall {overall:.2f}{marker}\n")

    meta = agent.checkpoint(seen_levels, best_score, sum(recent) / len(recent) if recent else 0.0)
    path = agent.model.save(meta, checkpoint_name)

    if not quiet:
        elapsed = time.perf_counter() - started
        print(f"\ntrained {episodes} episodes in {elapsed:.1f}s ({episodes/elapsed:.1f} ep/s)")
        print(f"saved final weights to {path.name}: {meta.describe()}")

        final_reports = evaluate(agent, levels)
        final_overall = sum(report.mean for report in final_reports) / len(final_reports)
        print("\n-- final weights --")
        print(summarise(final_reports))

        # A run shorter than EVAL_EVERY never reaches a periodic evaluation, which
        # would otherwise leave no best checkpoint at all.
        if final_overall > best_eval:
            best_eval, best_episode = final_overall, episodes
            agent.model.save(agent.checkpoint(seen_levels, best_score, final_overall), best_name)

        print(f"\n-- best weights ({best_name}, overall {best_eval:.2f} at episode {best_episode}) --")
        if final_overall < best_eval:
            print(
                f"final weights score {final_overall:.2f}, below the peak of {best_eval:.2f}. "
                f"Use {best_name} — this is the late-training collapse the target "
                "network is meant to limit, not a reason to trust the last epoch."
            )
        if eval_history:
            trail = "  ".join(f"{ep}:{score:.0f}" for ep, score in eval_history[-8:])
            print(f"eval trail  {trail}")

    return agent, meta


__all__ = ["train", "FEATURE_VERSION"]
