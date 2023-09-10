import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import SnakeGame and other necessary classes from snake_game.py
from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE, SnakeGameAI

# Define your SnakeAI model here
class SnakeAI(nn.Module):
    def __init__(self, input_size, num_actions):
        super(SnakeAI, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define your training function here
def train_snake_ai(model, game, num_episodes):
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    gamma = 0.9  # Discount factor

    for episode in range(num_episodes):
        game.reset()
        state = game.get_state()
        done = False

        while not done:
            # Convert the game state to a tensor
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)

            # Use the model to predict the action (e.g., up, down, left, right)
            action_values = model(state_tensor)
            action = torch.argmax(action_values).item()

            # Execute the action and get the reward and next state from the game
            reward, done, next_state = game.play_step(action)

            # Convert the next state to a tensor
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)

            # Calculate the target Q-value using the Bellman equation
            target_q = reward + gamma * torch.max(model(next_state_tensor))

            # Calculate the predicted Q-value for the current state
            predicted_q = model(state_tensor).gather(1, torch.tensor([[action]]))

            # Calculate the loss
            loss = criterion(predicted_q, target_q.unsqueeze(0))

            # Perform a gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the current state
            state = next_state

        print(f"Episode {episode + 1} completed with score: {game.score}")

    # Save the trained model
    torch.save(model.state_dict(), 'snake_ai_model.pth')

def main():
    # Initialize the game environment
    game = SnakeGameAI()

    # Initialize the AI model
    input_size = 3
    num_actions = 4
    ai_model = SnakeAI(input_size, num_actions)

    # Start training
    train_snake_ai(ai_model, game, num_episodes=1000)

if __name__ == "__main__":
    main()
