import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random

from snake_env import SnakeEnv
from dqn_agent import DQN

WIDTH = 1080
HEIGHT = 720
SIZE = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = SnakeEnv(WIDTH, HEIGHT, SIZE)

input_shape = env.get_state().shape
num_actions = 4

policy_net = DQN(input_shape, num_actions).to(device)
target_net = DQN(input_shape, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
memory = deque(maxlen=100000)


def select_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    else:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1).item()


def optimize_model(batch_size):
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.cat(state_batch)
    action_batch = torch.tensor(action_batch, device=device).unsqueeze(1)
    reward_batch = torch.tensor(reward_batch, device=device)
    next_state_batch = torch.cat(next_state_batch)
    done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device)

    q_values = policy_net(state_batch).gather(1, action_batch).squeeze()
    next_q_values = target_net(next_state_batch).max(1)[0].detach()

    target_q_values = reward_batch + 0.99 * next_q_values * (1 - done_batch)

    loss = nn.SmoothL1Loss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    model.load_state_dict(torch.load(filename))


def train():
    num_episodes = 100000
    episode_durations = []
    batch_size = 32
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 5000

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / epsilon_decay)
        if episode % 100 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            save_model(policy_net, f"snake_model_episode_{episode}.pth")

        steps = 0
        while True:
            action = select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            memory.append((state, action, reward, next_state, done))

            state = next_state

            optimize_model(batch_size)

            env.render()  # Add this line to visualize the game during training

            steps += 1

            if done:
                episode_durations.append(steps)
                break

        if episode % 10 == 0:
            print(f"Episode: {episode}, Steps: {steps}, Epsilon: {epsilon}")

        if episode % 100 == 0:
            target_net.load_state_dict(policy_net.state_dict())


if __name__ == "__main__":
    input_shape = (1, HEIGHT // SIZE, WIDTH // SIZE)
    num_actions = 4  # Number of possible actions in the Snake game
    policy_net = DQN(input_shape, num_actions, device, WIDTH, HEIGHT, SIZE).to(device)
    target_net = DQN(input_shape, num_actions, device, WIDTH, HEIGHT, SIZE).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Load a saved model if needed
    load_model(policy_net, "snake_model_episode_X.pth")
    load_model(target_net, "snake_model_episode_X.pth")

    train()
