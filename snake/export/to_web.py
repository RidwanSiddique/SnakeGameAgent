"""Export a trained checkpoint to the web client.

Writes web/public/agent/weights.json — plain nested arrays the browser applies as
two matrix multiplies. The feature version travels with the weights so the client
can refuse a mismatch loudly instead of misreading every input.

Usage: python3 -m snake.export.to_web [--name agent_best.pth]
"""
import argparse
import json
from pathlib import Path

from ..core.state import FEATURE_VERSION
from ..train.model import Linear_QNet

WEB_WEIGHTS = Path(__file__).resolve().parents[2] / "web" / "public" / "agent" / "weights.json"


def export(checkpoint_name: str = "agent_best.pth", destination: Path | None = None) -> Path:
    destination = destination or WEB_WEIGHTS
    model, meta = Linear_QNet.load(checkpoint_name, expect_feature_version=FEATURE_VERSION)

    path = model.export_weights(destination)

    # Stamp the feature version and provenance into the payload after the fact,
    # so the client can verify compatibility and the page can show what it loaded.
    payload = json.loads(path.read_text())
    payload["feature_version"] = meta.feature_version
    payload["trained"] = {
        "games": meta.games,
        "levels": meta.levels_trained,
        "best_score": meta.best_score,
        "mean_score": round(meta.mean_score, 2),
    }
    path.write_text(json.dumps(payload))
    return path


def main():
    parser = argparse.ArgumentParser(description="Export trained weights to the web client.")
    parser.add_argument("--name", default="agent_best.pth")
    args = parser.parse_args()

    path = export(args.name)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path} ({size_kb:.1f} KB)")
    if size_kb > 100:
        print("warning: over the 100KB budget for a browser payload")


if __name__ == "__main__":
    main()
