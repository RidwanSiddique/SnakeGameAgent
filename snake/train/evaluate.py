"""Measuring how good an agent actually is.

Training scores are a poor gauge: they are collected while the agent is still
exploring randomly, and they mix levels together. Evaluation runs greedily on
fixed seeds, so two runs of the same checkpoint produce the same number and "did
that change help?" stops being a matter of opinion.
"""
from dataclasses import dataclass
from statistics import mean, median

from ..core.engine import SnakeEngine
from ..core.state import get_state
from ..core.types import Action

DEFAULT_EPISODES = 30
# Hard cap per episode so a defensive agent that never dies cannot hang a run.
MAX_STEPS = 5_000


@dataclass
class LevelReport:
    level_id: int
    level_name: str
    scores: list[int]
    deaths_by_stall: int

    @property
    def mean(self) -> float:
        return mean(self.scores) if self.scores else 0.0

    @property
    def median(self) -> float:
        return median(self.scores) if self.scores else 0.0

    @property
    def best(self) -> int:
        return max(self.scores) if self.scores else 0

    def line(self) -> str:
        return (
            f"L{self.level_id} {self.level_name:<18} "
            f"mean {self.mean:6.2f}   median {self.median:5.1f}   best {self.best:3d}"
        )


def play_episode(agent, engine, *, max_steps: int = MAX_STEPS) -> tuple[int, bool]:
    """Play one greedy episode. Returns (score, ended_by_step_limit)."""
    for _ in range(max_steps):
        action = agent.get_action(get_state(engine), greedy=True)
        if engine.step(action).died:
            return engine.score, False
    return engine.score, True


def evaluate(agent, levels, *, episodes: int = DEFAULT_EPISODES, seed_base: int = 10_000):
    """Evaluate on every level over a fixed set of seeds."""
    reports = []
    for level_id in sorted(levels):
        level = levels[level_id]
        scores, stalls = [], 0
        for index in range(episodes):
            # Seeds are derived from the level so each board is exercised
            # consistently, and from a base far from training seeds.
            engine = SnakeEngine(level, seed=seed_base + level_id * 1000 + index)
            score, stalled = play_episode(agent, engine)
            scores.append(score)
            stalls += int(stalled)
        reports.append(LevelReport(level_id, level.name, scores, stalls))
    return reports


def summarise(reports) -> str:
    lines = [report.line() for report in reports]
    overall = mean([report.mean for report in reports]) if reports else 0.0
    lines.append(f"{'':<22} overall mean {overall:.2f}")
    return "\n".join(lines)
