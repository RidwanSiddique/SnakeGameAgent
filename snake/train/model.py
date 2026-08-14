"""Q-network and its trainer.

Two changes from the original model.py, both consequential:

**Checkpoints carry identity.** The old `save()` always wrote `model/model.pth`,
so training level 2 silently destroyed the level 1 model and the surviving file
could not be attributed to anything. Checkpoints are now named and carry metadata
— feature version, architecture, what it was trained on, how it scored. Loading a
checkpoint whose feature version disagrees with the current encoder raises rather
than quietly misinterpreting its inputs.

**Batch training is vectorised.** The old `train_step` looped over the batch and
called the network once per sample, so a batch of 1000 cost 1000 forward passes.
It now costs one.
"""
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


@dataclass
class Checkpoint:
    """Everything needed to identify and reload a trained network."""

    feature_version: int
    feature_count: int
    hidden_size: int
    output_size: int
    levels_trained: list[int] = field(default_factory=list)
    games: int = 0
    best_score: int = 0
    mean_score: float = 0.0
    saved_at: float = field(default_factory=time.time)

    def describe(self) -> str:
        levels = ", ".join(str(level) for level in self.levels_trained) or "none"
        return (
            f"features v{self.feature_version} ({self.feature_count} inputs), "
            f"hidden {self.hidden_size}, levels [{levels}], "
            f"{self.games} games, best {self.best_score}, mean {self.mean_score:.2f}"
        )


class Linear_QNet(nn.Module):
    """Feedforward Q-network. Name kept from the original model.py."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)

    def save(self, checkpoint: Checkpoint, file_name: str = "agent.pth") -> Path:
        """Write weights and metadata together, so the pair cannot drift apart."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / file_name
        torch.save({"state_dict": self.state_dict(), "meta": asdict(checkpoint)}, path)
        return path

    @classmethod
    def load(cls, file_name: str = "agent.pth", *, expect_feature_version: int | None = None):
        """Load a checkpoint, refusing one that was trained on other features."""
        path = MODEL_DIR / file_name
        if not path.exists():
            raise FileNotFoundError(f"no checkpoint at {path}")

        blob = torch.load(path, map_location="cpu", weights_only=False)
        if "meta" not in blob:
            raise ValueError(
                f"{path.name} predates checkpoint metadata and cannot be identified. "
                "It was saved by the original model.py, whose 14-feature inputs are "
                "incompatible with the current encoder. Retrain rather than load it."
            )

        meta = Checkpoint(**blob["meta"])
        if expect_feature_version is not None and meta.feature_version != expect_feature_version:
            raise ValueError(
                f"{path.name} was trained on feature version {meta.feature_version}, "
                f"but the encoder is version {expect_feature_version}. Its inputs would "
                "mean something different. Retrain or keep the matching encoder."
            )

        model = cls(meta.feature_count, meta.hidden_size, meta.output_size)
        model.load_state_dict(blob["state_dict"])
        return model, meta

    def export_weights(self, path: Path) -> Path:
        """Dump weights as JSON for the browser.

        The web client runs this network as two matrix multiplies, so it needs
        plain nested lists rather than a torch archive.
        """
        payload = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "layers": [
                {
                    "weight": self.net[0].weight.detach().tolist(),
                    "bias": self.net[0].bias.detach().tolist(),
                    "activation": "relu",
                },
                {
                    "weight": self.net[2].weight.detach().tolist(),
                    "bias": self.net[2].bias.detach().tolist(),
                    "activation": "linear",
                },
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path


class QTrainer:
    """Deep Q-learning update."""

    def __init__(self, model: Linear_QNet, lr: float, gamma: float):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        """One gradient step over a batch (or a single transition).

        `actions` are action indices, not one-hot vectors: the network output is
        indexed directly, which removes the per-sample argmax the original did.
        """
        states = torch.as_tensor(states, dtype=torch.float)
        next_states = torch.as_tensor(next_states, dtype=torch.float)
        actions = torch.as_tensor(actions, dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float)
        dones = torch.as_tensor(dones, dtype=torch.bool)

        if states.dim() == 1:  # single transition
            states = states.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
            actions = actions.reshape(1)
            rewards = rewards.reshape(1)
            dones = dones.reshape(1)

        # Q(s, a) for the actions actually taken.
        predicted = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target. No gradient flows through the bootstrap, and terminal
        # states take their reward alone.
        with torch.no_grad():
            best_next = self.model(next_states).max(dim=1).values
            target = rewards + self.gamma * best_next * (~dones)

        self.optimizer.zero_grad()
        loss = self.criterion(predicted, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
