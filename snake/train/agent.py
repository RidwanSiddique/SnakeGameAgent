"""The learning agent: action selection, replay memory, and reward shaping.

Reward shaping lives here rather than in the engine. That is what keeps the
engine portable to TypeScript, and it fixes a real defect in the original: the
old `agent.last_distance` was never reset between episodes, so the first step of
every game was shaped against the final step of the previous one.
"""
import random
from collections import deque

import numpy as np
import torch

from ..core.state import FEATURE_COUNT, FEATURE_VERSION, get_state, path_to_food
from ..core.types import Action
from .model import Checkpoint, Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256
GAMMA = 0.9

# Learn from a sampled minibatch every REPLAY_EVERY steps, once WARMUP_STEPS of
# experience exist. The original structure ran a gradient step on a *single*
# transition every step — over a thousand batch-of-one updates per episode — and
# measurably stopped improving after ~250 episodes while evaluation scores swung
# between 19 and 53. Sampled minibatches decorrelate consecutive frames, which is
# the entire point of a replay buffer.
REPLAY_EVERY = 4
WARMUP_STEPS = 2_000

# Exploration decays from EPS_START to EPS_END over EPS_DECAY_GAMES episodes. The
# floor stays off zero: with four levels sharing one network, a purely greedy
# agent stops discovering anything new about the levels it currently plays worst.
EPS_START = 0.90
EPS_END = 0.05
EPS_DECAY_GAMES = 1_500

# Reward terms. Eating and dying dominate; the distance terms only break ties so
# the agent does not wander early in training while it has no idea where food is.
REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
REWARD_CLOSER = 0.10
REWARD_FARTHER = -0.15  # slightly worse than closer is good, to discourage dithering


class Agent:
    """Deep Q-learning agent over the 21-feature state encoding."""

    def __init__(self, model: Linear_QNet | None = None):
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = model or Linear_QNet(FEATURE_COUNT, HIDDEN_SIZE, len(Action))
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=GAMMA)
        self._prev_distance = None

    # --- perception -----------------------------------------------------------

    @staticmethod
    def get_state(engine) -> np.ndarray:
        return get_state(engine)

    # --- acting ---------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.n_games / EPS_DECAY_GAMES)
        return EPS_START + (EPS_END - EPS_START) * progress

    def get_action(self, state, *, greedy: bool = False) -> Action:
        """Epsilon-greedy action choice.

        Random exploration avoids moves the state vector already flags as fatal.
        Without that filter most early episodes end in a handful of steps and the
        replay buffer fills with near-identical deaths.
        """
        if not greedy and random.random() < self.epsilon:
            safe = [index for index in range(3) if state[index] == 0]
            return Action(random.choice(safe) if safe else random.randrange(3))

        with torch.no_grad():
            q_values = self.model(torch.as_tensor(state, dtype=torch.float))
        return Action(int(torch.argmax(q_values).item()))

    # --- reward ---------------------------------------------------------------

    def start_episode(self, engine) -> None:
        """Reset per-episode shaping state. Skipping this is the old bug."""
        self._prev_distance = self._food_distance(engine)

    @staticmethod
    def _food_distance(engine) -> int | None:
        """Length of the shortest *route* to the food, not the straight line.

        Manhattan distance was actively harmful on obstacle-dense boards: going
        around a wall increases it, so every step of a correct detour was
        penalised. Measured on the Corridors level, that produced an agent which
        pressed toward walls and died to the stall timer in 100% of evaluation
        episodes without ever eating. Path distance rewards the detour instead.
        """
        if engine.food is None:
            return None
        _, distance = path_to_food(engine)
        return distance

    def reward_for(self, engine, result) -> float:
        """Turn an engine result into a scalar reward."""
        if result.died:
            self._prev_distance = None
            return REWARD_DEATH

        if result.ate:
            # Food moved, so the old distance is meaningless; re-anchor.
            self._prev_distance = self._food_distance(engine)
            return REWARD_FOOD

        distance = self._food_distance(engine)
        reward = 0.0
        if distance is not None and self._prev_distance is not None:
            reward = REWARD_CLOSER if distance < self._prev_distance else REWARD_FARTHER
        self._prev_distance = distance
        return reward

    # --- learning -------------------------------------------------------------

    def remember(self, state, action: Action, reward, next_state, done) -> None:
        self.memory.append((state, action.value, reward, next_state, done))

    def replay(self, batch_size: int = BATCH_SIZE):
        """Train on a minibatch sampled from replay memory.

        Returns None until WARMUP_STEPS of experience exist, so early updates are
        not dominated by the handful of transitions collected so far.
        """
        if len(self.memory) < WARMUP_STEPS:
            return None
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return self.trainer.train_step(
            np.array(states), actions, rewards, np.array(next_states), dones
        )

    # --- persistence ----------------------------------------------------------

    def checkpoint(self, levels_trained, best_score: int, mean_score: float) -> Checkpoint:
        return Checkpoint(
            feature_version=FEATURE_VERSION,
            feature_count=FEATURE_COUNT,
            hidden_size=HIDDEN_SIZE,
            output_size=len(Action),
            levels_trained=sorted(levels_trained),
            games=self.n_games,
            best_score=best_score,
            mean_score=mean_score,
        )
