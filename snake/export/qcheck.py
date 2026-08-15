"""Record Python's Q-values so the browser's forward pass can be verified.

The engine parity suite is deliberately weight-independent, so it would not
catch a bug in weight export, base64 decoding, or matrix layout — any of which
would make the deployed agent choose different moves from the trained one. This
fixture closes that gap.
"""
import json
from pathlib import Path

import torch

from ..core.engine import SnakeEngine
from ..core.levels import load_levels
from ..core.state import FEATURE_VERSION, get_state
from ..core.types import Action
from ..train.model import Linear_QNet

OUTPUT = Path(__file__).resolve().parents[2] / "shared" / "golden" / "qvalues.json"


def record(checkpoint_name: str = "agent_best.pth") -> Path:
    model, meta = Linear_QNet.load(checkpoint_name, expect_feature_version=FEATURE_VERSION)
    model.eval()
    levels = load_levels()

    samples = []
    for level_id in sorted(levels):
        engine = SnakeEngine(levels[level_id], seed=700 + level_id)
        for _ in range(25):
            state = get_state(engine)
            with torch.no_grad():
                q = model(torch.as_tensor(state, dtype=torch.float))
            samples.append({
                "level_id": level_id,
                "state": [round(float(v), 6) for v in state.tolist()],
                "q": [round(float(v), 5) for v in q.tolist()],
            })
            if engine.step(Action(int(torch.argmax(q).item()))).died:
                engine.reset()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"feature_version": FEATURE_VERSION, "samples": samples}))
    return OUTPUT


if __name__ == "__main__":
    path = record()
    print(f"wrote {path} — {len(json.loads(path.read_text())['samples'])} samples")
