"""Recording reference trajectories for the TypeScript port to reproduce.

This is the enforcement half of the parity contract. Python plays a fixed level
with a fixed seed and a fixed action sequence, recording every frame; the web
test suite replays the same inputs through its own engine and asserts the frames
match. Any divergence — a different food cell, an off-by-one wall, a reordered
feature — fails there rather than showing up as an agent that mysteriously plays
worse in the browser than it did in training.

Frames record engine state *and* the full feature vector, because the encoder is
as much a part of the contract as the rules are.
"""
import json
from pathlib import Path

from ..core.engine import SnakeEngine
from ..core.levels import load_levels
from ..core.rng import Rng
from ..core.state import FEATURE_COUNT, FEATURE_VERSION, get_state
from ..core.types import Action

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "shared" / "golden"

# Action sequences are generated from the shared RNG rather than a policy, so
# reproducing them needs no model — the web test suite stays weight-independent.
EPISODES_PER_LEVEL = 3
MAX_FRAMES = 500


def record_episode(level, seed: int, action_seed: int, max_frames: int = MAX_FRAMES) -> dict:
    """Play one scripted episode and capture every frame."""
    engine = SnakeEngine(level, seed=seed)
    action_rng = Rng(action_seed)

    frames = [{**engine.snapshot(), "state": [round(v, 6) for v in get_state(engine).tolist()]}]
    actions = []

    for _ in range(max_frames):
        action = Action(action_rng.randint(0, 2))
        actions.append(action.value)
        result = engine.step(action)
        frames.append(
            {
                **engine.snapshot(),
                "state": [round(v, 6) for v in get_state(engine).tolist()],
                "ate": result.ate,
                "died": result.died,
            }
        )
        if result.died:
            break

    return {
        "level_id": level.id,
        "level_name": level.name,
        "seed": seed,
        "action_seed": action_seed,
        "actions": actions,
        "obstacles": sorted([cell.x, cell.y] for cell in engine.obstacles),
        "frames": frames,
    }


def record_all(output_dir: Path | None = None) -> Path:
    """Record the full golden set and write it to shared/golden/."""
    output_dir = output_dir or GOLDEN_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = load_levels()
    episodes = []
    for level_id in sorted(levels):
        for index in range(EPISODES_PER_LEVEL):
            episodes.append(
                record_episode(levels[level_id], seed=500 + index, action_seed=900 + index)
            )

    payload = {
        "feature_version": FEATURE_VERSION,
        "feature_count": FEATURE_COUNT,
        "rng": "xorshift128, splitmix32 seeding — see snake/core/rng.py",
        "episodes": episodes,
    }

    path = output_dir / "trajectories.json"
    path.write_text(json.dumps(payload))
    return path


if __name__ == "__main__":
    written = record_all()
    blob = json.loads(written.read_text())
    frames = sum(len(episode["frames"]) for episode in blob["episodes"])
    print(f"wrote {written} — {len(blob['episodes'])} episodes, {frames} frames")
