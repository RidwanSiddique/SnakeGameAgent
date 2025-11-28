import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize Pygame
pygame.init()

# Set up fonts
font = pygame.font.Font('arial.ttf', 25)

# Direction enum to represent movement
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Named tuple to represent a point (x, y)
Point = namedtuple('Point', 'x, y')

# RGB color definitions
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)  # Color for obstacles

# Game configuration
BLOCK_SIZE = 20  # Size of each block in the game
SPEED = 40  # Game speed (frames per second)

class SnakeGameAI:
    """
    Snake game environment for AI, designed for reinforcement learning.
    Supports multiple difficulty levels.
    """
    def __init__(self, width=640, height=480, level=1):
        """
        Initialize the game with the given width and height.
        :param width: Width of the game window.
        :param height: Height of the game window.
        :param level: Game level (1 for blank, 2 for obstacles).
        """
        self.width = width
        self.height = height
        self.level = level

        # Initialize the game display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f'Snake - Level {level}')

        # Initialize the game clock
        self.clock = pygame.time.Clock()

        # Initialize obstacles
        self.obstacles = []

        # Reset the game state
        self.reset()

    def reset(self):
        """
        Reset the game to the initial state.
        """
        self.direction = Direction.RIGHT  # Initial direction of the snake
        self.head = Point(self.width / 2, self.height / 2)  # Snake's initial position
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]  # Initial snake body

        self.score = 0  # Initial score
        self.food = None  # Food position
        self.obstacles = []  # Clear obstacles

        # Generate obstacles based on level
        if self.level == 2:
            self._generate_obstacles_level2()

        self._place_food()  # Place the first food item
        self.frame_iteration = 0  # Frame counter to detect stalls

    def _generate_obstacles_level2(self):
        """
        Generate obstacles for Level 2.
        Uses a fixed layout for consistent training across sessions.
        """
        # Fixed obstacle layout for consistent training
        obstacles = [
            # Large 3x3 block in top-left area
            Point(100, 60), Point(120, 60), Point(140, 60),
            Point(100, 80), Point(120, 80), Point(140, 80),
            Point(100, 100), Point(120, 100), Point(140, 100),
            
            # Large 3x3 block in bottom-right area
            Point(460, 320), Point(480, 320), Point(500, 320),
            Point(460, 340), Point(480, 340), Point(500, 340),
            Point(460, 360), Point(480, 360), Point(500, 360),
            
            # 2x2 blocks scattered
            Point(200, 180), Point(220, 180),
            Point(200, 200), Point(220, 200),
            
            Point(400, 140), Point(420, 140),
            Point(400, 160), Point(420, 160),
            
            Point(300, 300), Point(320, 300),
            Point(300, 320), Point(320, 320),
            
            # Single obstacles for navigation challenges
            Point(160, 240), Point(260, 120), Point(380, 220),
            Point(180, 320), Point(340, 180), Point(520, 200),
            Point(140, 380), Point(480, 260), Point(300, 60)
        ]
        
        # Filter obstacles that are too close to starting position
        safe_obstacles = []
        for obstacle in obstacles:
            if (abs(obstacle.x - self.head.x) > 80 or abs(obstacle.y - self.head.y) > 80):
                safe_obstacles.append(obstacle)
        
        self.obstacles = safe_obstacles

    def _place_food(self):
        """
        Place food at a random position on the game board.
        Ensures food doesn't spawn on snake or obstacles.
        """
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)

            # Ensure food does not spawn on the snake or obstacles
            if self.food not in self.snake and self.food not in self.obstacles:
                return
            
            attempts += 1
        
        # Fallback: if we can't find a spot after many attempts, place it anywhere not on snake
        while True:
            x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                return

    def play_step(self, action):
        """
        Execute one step of the game based on the given action.
        :param action: Action taken by the AI [straight, right, left].
        :return: reward, game_over (bool), score (int)
        """
        self.frame_iteration += 1

        # Handle user input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move the snake
        self._move(action)  # Update the snake's head position
        self.snake.insert(0, self.head)

        # Check for collisions or game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Check if the snake eats food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()  # Remove the tail if no food is eaten

        # Update the game UI and control the frame rate
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_collision(self, point=None):
        """
        Check if the given point collides with the wall, obstacles, or the snake's body.
        :param point: The point to check for collisions (defaults to snake's head).
        :return: True if there is a collision, False otherwise.
        """
        if point is None:
            point = self.head

        # Check if the point is outside the game boundaries
        if point.x >= self.width or point.x < 0 or point.y >= self.height or point.y < 0:
            return True

        # Check if the point collides with the snake's body
        if point in self.snake[1:]:
            return True

        # Check if the point collides with obstacles (Level 2)
        if point in self.obstacles:
            return True

        return False

    def _update_ui(self):
        """
        Update the game UI with the current state.
        """
        self.display.fill(BLACK)  # Fill the background with black

        # Draw obstacles (Level 2)
        for obstacle in self.obstacles:
            pygame.draw.rect(self.display, GRAY, pygame.Rect(obstacle.x, obstacle.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw the snake
        for point in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(point.x + 4, point.y + 4, 12, 12))

        # Draw the food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Display the score and level
        score_text = font.render(f"Score: {self.score} | Level: {self.level}", True, WHITE)
        self.display.blit(score_text, [0, 0])

        # Refresh the display
        pygame.display.flip()

    def _move(self, action):
        """
        Move the snake based on the given action.
        :param action: Action taken by the AI [straight, right, left].
        """
        # Possible directions in clockwise order
        directions_clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_idx = directions_clockwise.index(self.direction)

        # Determine the new direction based on the action
        if np.array_equal(action, [1, 0, 0]):
            # No change in direction
            new_direction = directions_clockwise[current_idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Turn right
            next_idx = (current_idx + 1) % 4
            new_direction = directions_clockwise[next_idx]
        else:  # [0, 0, 1]
            # Turn left
            next_idx = (current_idx - 1) % 4
            new_direction = directions_clockwise[next_idx]

        self.direction = new_direction

        # Update the snake's head position based on the new direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
