# Machine Learning Powered Snake Game

## Overview
This project combines the classic Snake game, implemented in PyGame, with a machine learning agent built using TensorFlow. The goal is to demonstrate the capabilities of machine learning techniques in learning and mastering gameplay. The agent is trained to navigate the game environment, avoid obstacles, and maximize its score by eating food items.

## Features
- **Snake Game Implementation:** A complete Snake game built with PyGame for real-time gameplay.
- **TensorFlow Agent:** A machine learning model designed to learn and play the Snake game autonomously.
- **Training and Evaluation:** Tools and scripts for training the model and evaluating its performance over time.

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

