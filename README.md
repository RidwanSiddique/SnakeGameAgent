# Machine Learning Powered Snake Game

## Overview
This project marries the nostalgic appeal of the classic Snake game with the cutting-edge field of machine learning, providing an engaging showcase of AI's potential in game development. Utilizing PyGame for game implementation and TensorFlow for crafting a learning agent, this initiative demonstrates a sophisticated approach to game playing through artificial intelligence. The core of the project lies in its machine learning agent, which learns to navigate the game's challenges—avoiding collisions with walls and its tail while pursuing food to increase its length and score. Through continuous interaction with the game environment, the agent employs reinforcement learning to improve its strategy over time, embodying the principles of trial and error and reward-based learning. This project not only highlights the practical application of machine learning in understanding and mastering complex tasks but also serves as an accessible entry point for enthusiasts looking to explore the integration of AI with game development.


## Preview
https://github.com/RidwanSiddique/SnakeGameAgent/assets/65805850/acdf8373-98e8-4940-b57a-69a6b54bc634


## Features
- **Dynamic Snake Game Implementation:** At the heart of this project is a fully functional Snake game developed using PyGame. This implementation captures all the quintessential elements of the original game, including the snake that navigates through the screen, grows in length with each piece of food consumed, and the game-over scenario triggered by colliding with the walls or itself. The game's fluid controls and real-time response mechanics provide a seamless and immersive gaming experience.

- **Advanced TensorFlow Agent:** The project's crown jewel is its TensorFlow-based machine learning agent. This agent is designed with a neural network architecture that enables it to perceive the game environment, make decisions, and learn from the outcomes of those decisions. Using reinforcement learning, the agent gradually optimizes its strategy to increase game longevity and maximize scores. This not only demonstrates the agent's ability to understand complex patterns and adapt its behavior but also showcases TensorFlow's versatility in developing sophisticated AI models.

- **Comprehensive Training and Evaluation Framework:** A robust suite of tools and scripts accompanies the project to facilitate the training, testing, and evaluation of the machine learning agent. This framework allows for the monitoring of the agent's performance over multiple iterations, providing insights into its learning progression and areas for improvement. Through detailed analytics and visualizations, developers can tweak the model's parameters, refine its architecture, and experiment with different learning algorithms to enhance the agent's gameplay prowess.


## Getting Started

### Prerequisites
Before running this project, you need to have Conda installed on your machine. Conda will be used to create an isolated environment for the project's dependencies.

### Environment Setup
To set up the project environment, follow these steps in your terminal:

1. **Create Conda Environment:** Create a new Conda environment named `pygame_env`.
    ```shell
    conda create -n pygame_env
    ```
2. **Activate Environment:** Activate the newly created environment.
    ```shell
    conda activate pygame_env
    ```
3. **Install Dependencies:** Install the required Python packages including PyGame, PyTorch, and other utilities.
    ```shell
    pip install pygame
    pip install torch torchvision
    pip install matplotlib ipython
    ```

### Running the Game
Once the environment is set up and all dependencies are installed, you can start the game and the machine learning agent by running the main script. Ensure you are in the project's root directory and the `pygame_env` is activated.
```shell
python agent.py

