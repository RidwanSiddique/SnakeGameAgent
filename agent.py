import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# Configuration constants
MAX_MEMORY = 100_000  # Maximum memory for experience replay
BATCH_SIZE = 1000  # Batch size for training
LEARNING_RATE = 0.001  # Learning rate for the optimizer


class Agent:
    """
    Reinforcement Learning agent for Snake AI, using Deep Q-Learning.
    """
    def __init__(self):
        """
        Initialize the agent with the required components.
        """
        self.n_games = 0  # Counter for the number of games played
        self.epsilon = 0  # Exploration rate (higher at start, reduces over time)
        self.gamma = 0.9  # Discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Replay memory (FIFO queue)
        self.model = Linear_QNet(11, 256, 3)  # Q-network with input, hidden, and output layers
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)  # Trainer for the model

    def get_state(self, game):
        """
        Compute the current state of the game as a feature vector.

        :param game: Instance of the SnakeGameAI class.
        :return: Numpy array representing the state of the game.
        """
        # Snake's current position (head)
        head = game.snake[0]
        # Adjacent positions around the snake's head
        point_left = Point(head.x - 20, head.y)
        point_right = Point(head.x + 20, head.y)
        point_up = Point(head.x, head.y - 20)
        point_down = Point(head.x, head.y + 20)

        # Current movement direction
        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        # Game state representation
        state = [
            # Danger in the current direction
            (direction_right and game.is_collision(point_right)) or
            (direction_left and game.is_collision(point_left)) or
            (direction_up and game.is_collision(point_up)) or
            (direction_down and game.is_collision(point_down)),

            # Danger if turning right
            (direction_up and game.is_collision(point_right)) or
            (direction_down and game.is_collision(point_left)) or
            (direction_left and game.is_collision(point_up)) or
            (direction_right and game.is_collision(point_down)),

            # Danger if turning left
            (direction_down and game.is_collision(point_right)) or
            (direction_up and game.is_collision(point_left)) or
            (direction_right and game.is_collision(point_up)) or
            (direction_left and game.is_collision(point_down)),

            # Movement direction (one-hot encoding)
            direction_left,
            direction_right,
            direction_up,
            direction_down,

            # Food position relative to the snake
            game.food.x < game.head.x,  # Food is to the left
            game.food.x > game.head.x,  # Food is to the right
            game.food.y < game.head.y,  # Food is above
            game.food.y > game.head.y   # Food is below
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """
        Store a single experience tuple in memory.

        :param state: Current state.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state after action.
        :param done: Boolean indicating if the game is over.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Train the model on a batch of experiences from memory.
        """
        # Sample a mini-batch from memory
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_batch = self.memory

        # Unpack the mini-batch
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the model on a single experience tuple.

        :param state: Current state.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state after action.
        :param done: Boolean indicating if the game is over.
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Decide the next action using an epsilon-greedy policy.

        :param state: Current state of the game.
        :return: Action to be performed (one-hot encoded).
        """
        self.epsilon = 80 - self.n_games  # Reduce exploration rate as games progress
        final_move = [0, 0, 0]  # Initialize move vector

        if random.randint(0, 200) < self.epsilon:
            # Random move (exploration)
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Predicted move (exploitation)
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    """
    Main training loop for the Snake AI agent.
    """
    # Plotting variables
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record_score = 0

    # Initialize agent and game
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Get the current state
        state_old = agent.get_state(game)

        # Determine action
        final_move = agent.get_action(state_old)

        # Perform action and get the new state, reward, and game status
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train the agent on the current move
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Store the experience in memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Game over: Train on long-term memory and reset the game
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Save the model if a new record is achieved
            if score > record_score:
                record_score = score
                agent.model.save()

            # Log game results
            print(f'Game {agent.n_games}, Score: {score}, Record: {record_score}')

            # Update plotting variables
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
