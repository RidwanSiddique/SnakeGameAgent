import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    """
    A simple feedforward neural network for Q-learning, with one hidden layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the network layers.

        :param input_size: Number of input features.
        :param hidden_size: Number of neurons in the hidden layer.
        :param output_size: Number of output actions (Q-values).
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # Input to hidden layer
        self.linear2 = nn.Linear(hidden_size, output_size) # Hidden to output layer

    def forward(self, x):
        """
        Forward pass of the network.

        :param x: Input tensor.
        :return: Output tensor (Q-values for each action).
        """
        x = F.relu(self.linear1(x)) # Apply ReLU activation to the hidden layer
        x = self.linear2(x) # Compute the output layer
        return x

    def save(self, file_name='model.pth'):
        """
        Save the model's state dictionary to a file.

        :param file_name: The name of the file where the model is saved.
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path) # Create the directory if it doesn't exist

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name) # Save the model's parameters


class QTrainer:
    """
    A trainer class for training the Q-learning network.
    """
    def __init__(self, model, lr, gamma):
        """
        Initialize the trainer.

        :param model: The Q-learning model to be trained.
        :param learning_rate: Learning rate for the optimizer.
        :param discount_factor: Discount factor (gamma) for future rewards.
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model # The Q-network
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # Adam optimizer
        self.criterion = nn.MSELoss() # Mean Squared Error loss

    def train_step(self, state, action, reward, next_state, done):
        """
        Perform one training step of the Q-learning algorithm.

        :param state: Current state.
        :param action: Action taken at the current state.
        :param reward: Reward received after taking the action.
        :param next_state: Next state after taking the action.
        :param done: Boolean flag indicating if the episode is done.
        """
        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Reshape tensors to add batch dimension if only one sample is provided
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        # Clone the predicted Q-values to compute the target values
        target = pred.clone()
        
        # Update the target Q-values for each sample in the batch
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Bellman equation: Q_new = reward + gamma * max(Q(next_state))
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        # Perform gradient descent step
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()