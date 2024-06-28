# test_agent.py
import os

import numpy as np
from blackjack_env import BlackjackEnv

env = BlackjackEnv()

# Load the trained Q-table
if os.path.exists("q_table.npy"):
    Q = np.load("q_table.npy")
else:
    raise FileNotFoundError("Q-table not found. Train the model before testing.")


# Function to choose an action using the trained Q-table
def choose_action(state):
    return np.argmax(Q[state])


# Test the agent
num_episodes = 1000000
total_rewards = 0

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    # Random bet amount between $2 and 10% of player's money
    bet = min(max(2, env.game.player.money // 13), 500)
    env.game.player.place_bet(bet)

    while not done:
        action = choose_action(state)
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    total_rewards += episode_reward

average_reward = total_rewards / num_episodes
print(f"Average reward over {num_episodes} episodes: {average_reward}")
