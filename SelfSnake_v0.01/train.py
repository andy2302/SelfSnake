import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
import os
import math
from itertools import count

from snake_env import SnakeEnv
from dqn_agent import DQN

WIDTH = 1080
HEIGHT = 720
SIZE = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = SnakeEnv(WIDTH, HEIGHT, SIZE)

input_shape = env.get_state().shape
num_actions = 4

memory = deque(maxlen=100000)

num_episodes = 1000
batch_size = 128
TARGET_UPDATE = 10


def get_screen():
    state = env.get_state()
    state = torch.tensor(state, device=device).unsqueeze(0).unsqueeze(0)
    return state.float()


EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 200
steps_done = 0
epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1 * steps_done / EPSILON_DECAY)


def select_action(state, steps_done):
    global steps_episode
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1 * steps_done / EPSILON_DECAY)
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

    state_batch = torch.stack(state_batch)  # Use torch.stack instead of torch.cat
    action_batch = torch.tensor(action_batch, device=device).unsqueeze(1)
    reward_batch = torch.tensor(reward_batch, device=device)
    next_state_batch = torch.stack(next_state_batch)  # Use torch.stack instead of torch.cat
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
    global steps_done
    highest_score = 0
    moving_avg_len = 100
    scores = []
    moving_avg_scores = []

    for episode in range(num_episodes):
        env.reset()
        state = get_screen()
        score = 0

        for t in count():
            env.render()
            action = select_action(state, steps_done)
            steps_done += 1
            next_state, reward, done = env.step(action)
            score += reward

            if done:
                next_state = None

            memory.append((state, action, torch.tensor([reward], device=device), next_state, done))
            state = next_state if next_state is not None else state

            optimize_model(batch_size)

            if done:
                scores.append(score)
                if len(scores) > moving_avg_len:
                    moving_avg_scores.append(sum(scores[-moving_avg_len:]) / moving_avg_len)
                break

        highest_score = max(highest_score, score)

        # Display stats
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Episode: {episode + 1}/{num_episodes}")
        print(f"Current Score: {score}")
        print(f"Highest Score: {highest_score}")
        if len(scores) > moving_avg_len:
            print(f"Moving Average (last {moving_avg_len} episodes): {moving_avg_scores[-1]}")

        # Save the model every 100 episodes
        if (episode + 1) % 100 == 0:
            torch.save(policy_net.state_dict(), f"model_episode_{episode + 1}.pth")

        # Save the model when the highest score is achieved
        if score >= highest_score:
            torch.save(policy_net.state_dict(), "best_model.pth")

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("Training completed.")


if __name__ == "__main__":
    input_shape = (1, HEIGHT // SIZE, WIDTH // SIZE)
    num_actions = 4  # Number of possible actions in the Snake game
    policy_net = DQN(input_shape, num_actions, device, WIDTH, HEIGHT, SIZE).to(device)
    target_net = DQN(input_shape, num_actions, device, WIDTH, HEIGHT, SIZE).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)

    # Load a saved model if needed
    # load_model(policy_net, "snake_model_episode_X.pth")
    # load_model(target_net, "snake_model_episode_X.pth")

    train()
