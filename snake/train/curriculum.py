"""Which level to train on next.

One network plays every level, which the state encoding already permits: all 21
features are relative to the head, so nothing tells the agent which board it is
on. Training it on a *mix* is what makes that potential real — a network trained
only on level 2's fixed layout memorises that geometry, and the web level
designer would immediately expose it.

Stages shift weight from open boards toward obstacles as the agent gets
competent, but never drop a level entirely: the network has one set of weights,
and a level it stops seeing is a level it starts forgetting.
"""
from dataclasses import dataclass

# (games_played_threshold, {level_id: weight})
STAGES: list[tuple[int, dict[int, float]]] = [
    (0, {1: 1.00}),
    (100, {1: 0.50, 2: 0.30, 3: 0.20}),
    (300, {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}),
    (600, {1: 0.15, 2: 0.20, 3: 0.25, 4: 0.40}),
]


@dataclass
class Curriculum:
    """Samples a level id for each episode."""

    levels: dict
    stages: list[tuple[int, dict[int, float]]] = None

    def __post_init__(self):
        self.stages = self.stages or STAGES
        for _, weights in self.stages:
            unknown = set(weights) - set(self.levels)
            if unknown:
                raise ValueError(f"curriculum references unknown levels: {sorted(unknown)}")

    def weights_at(self, n_games: int) -> dict[int, float]:
        chosen = self.stages[0][1]
        for threshold, weights in self.stages:
            if n_games >= threshold:
                chosen = weights
        return chosen

    def sample(self, n_games: int, rng) -> int:
        """Pick a level id using the shared RNG, so runs stay reproducible."""
        weights = self.weights_at(n_games)
        ids = sorted(weights)
        total = sum(weights[i] for i in ids)

        # Draw a fixed-point fraction rather than a float, so this is portable
        # and does not depend on Python's float formatting.
        target = (rng.next_u32() / 2**32) * total
        cumulative = 0.0
        for level_id in ids:
            cumulative += weights[level_id]
            if target < cumulative:
                return level_id
        return ids[-1]

    def describe(self, n_games: int) -> str:
        weights = self.weights_at(n_games)
        return " ".join(f"L{i}:{weights[i]:.0%}" for i in sorted(weights))
