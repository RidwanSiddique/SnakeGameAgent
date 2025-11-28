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
        self.model = Linear_QNet(14, 256, 3)  # Q-network with input, hidden, and output layers (14 features for enhanced state)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)  # Trainer for the model

    def get_state(self, game):
        """
        Compute the current state of the game as a feature vector.
        Enhanced to handle obstacles in Level 2 and be robust for any level.

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

        # Enhanced danger detection - check immediate collision
        danger_straight = False
        danger_right = False  
        danger_left = False

        if direction_left:
            danger_straight = game.is_collision(point_left)
            danger_right = game.is_collision(point_up)
            danger_left = game.is_collision(point_down)
        elif direction_right:
            danger_straight = game.is_collision(point_right)
            danger_right = game.is_collision(point_down)
            danger_left = game.is_collision(point_up)
        elif direction_up:
            danger_straight = game.is_collision(point_up)
            danger_right = game.is_collision(point_right)
            danger_left = game.is_collision(point_left)
        elif direction_down:
            danger_straight = game.is_collision(point_down)
            danger_right = game.is_collision(point_left)
            danger_left = game.is_collision(point_right)

        # Look ahead for obstacles (enhanced for Level 2 large obstacles)
        def check_path_blocked(start_point, direction, steps=3):
            """Check if path is blocked within next few steps"""
            current = start_point
            for _ in range(steps):
                if direction == Direction.LEFT:
                    current = Point(current.x - 20, current.y)
                elif direction == Direction.RIGHT:
                    current = Point(current.x + 20, current.y)
                elif direction == Direction.UP:
                    current = Point(current.x, current.y - 20)
                elif direction == Direction.DOWN:
                    current = Point(current.x, current.y + 20)
                
                if game.is_collision(current):
                    return True
            return False

        # Enhanced path checking for better obstacle avoidance
        path_blocked_straight = False
        path_blocked_right = False
        path_blocked_left = False

        if direction_left:
            path_blocked_straight = check_path_blocked(head, Direction.LEFT)
            path_blocked_right = check_path_blocked(head, Direction.UP)
            path_blocked_left = check_path_blocked(head, Direction.DOWN)
        elif direction_right:
            path_blocked_straight = check_path_blocked(head, Direction.RIGHT)
            path_blocked_right = check_path_blocked(head, Direction.DOWN)
            path_blocked_left = check_path_blocked(head, Direction.UP)
        elif direction_up:
            path_blocked_straight = check_path_blocked(head, Direction.UP)
            path_blocked_right = check_path_blocked(head, Direction.RIGHT)
            path_blocked_left = check_path_blocked(head, Direction.LEFT)
        elif direction_down:
            path_blocked_straight = check_path_blocked(head, Direction.DOWN)
            path_blocked_right = check_path_blocked(head, Direction.LEFT)
            path_blocked_left = check_path_blocked(head, Direction.RIGHT)

        # Game state representation (enhanced for Level 2)
        state = [
            # Immediate danger
            danger_straight,
            danger_right,
            danger_left,

            # Path ahead blocked (helps with large obstacles)
            path_blocked_straight,
            path_blocked_right,
            path_blocked_left,

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
        Enhanced with better exploration strategy.
        :param state: Current state of the game.
        :return: Action to be performed (one-hot encoded).
        """
        # More gradual epsilon decay for better learning
        self.epsilon = max(5, 90 - self.n_games * 0.8)  # Slower decay, minimum 5%
        final_move = [0, 0, 0]  # Initialize move vector

        if random.randint(0, 100) < self.epsilon:
            # Smart random move (avoid immediate danger if possible)
            safe_moves = []
            if not state[0]:  # No danger straight
                safe_moves.append(0)
            if not state[1]:  # No danger right
                safe_moves.append(1)
            if not state[2]:  # No danger left
                safe_moves.append(2)
            
            if safe_moves:
                move = random.choice(safe_moves)
            else:
                move = random.randint(0, 2)  # No safe moves, pick randomly
            
            final_move[move] = 1
        else:
            # Predicted move (exploitation)
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(level=1):
    """
    Main training loop for the Snake AI agent.
    :param level: Game level to train on (1 for blank, 2 for obstacles).
    """
    # Plotting variables
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record_score = 0

    # Initialize agent and game
    agent = Agent()
    game = SnakeGameAI(level=level)

    while True:
        # Get the current state
        state_old = agent.get_state(game)

        # Determine action
        final_move = agent.get_action(state_old)

        # Perform action and get the new state, reward, and game status
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Enhanced reward system for better learning
        if not done:
            # Small positive reward for staying alive
            reward += 0.1
            
            # Distance-based reward towards food
            old_distance = abs(game.food.x - game.head.x) + abs(game.food.y - game.head.y)
            # Check if we're moving closer to food by comparing with previous distance
            if hasattr(agent, 'last_distance'):
                if old_distance < agent.last_distance:
                    reward += 1  # Moving closer to food
                elif old_distance > agent.last_distance:
                    reward -= 0.5  # Moving away from food
            agent.last_distance = old_distance

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
            print(f'Level {level} | Game {agent.n_games}, Score: {score}, Record: {record_score}')

            # Update plotting variables
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    import sys
    
    # Check if level is specified as command line argument
    level = 1
    if len(sys.argv) > 1:
        try:
            level = int(sys.argv[1])
            if level not in [1, 2]:
                print("Invalid level. Choose 1 or 2. Defaulting to level 1.")
                level = 1
        except ValueError:
            print("Invalid argument. Defaulting to level 1.")
            level = 1
    
    print(f"Starting training on Level {level}...")
    train(level=level)
