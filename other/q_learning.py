# q_learning.py
import numpy as np
from blackjack_env import BlackjackEnv
import matplotlib.pyplot as plt
import os

env = BlackjackEnv()

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000000
save_interval = 100000  # Save the Q-table every 10,000 episodes

# Q-table initialization
if os.path.exists("q_table.npy"):
    Q = np.load("q_table.npy")
else:
    Q = np.zeros((32, 11, 2, 3))


# Function to choose an action using epsilon-greedy policy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# Training the agent
rewards = []
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    # Random bet amount between $2 and 10% of player's money
    bet = min(max(2, env.game.player.money // 10), 500)
    env.game.player.place_bet(bet)

    while not done:
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        state = next_state
        total_reward += reward

    rewards.append(total_reward)

    if episode % save_interval == 0:
        np.save("q_table.npy", Q)
        print(f"Episode {episode}/{num_episodes} - Q-table saved")

# Save the Q-table after training
np.save("q_table.npy", Q)

# Plotting the results
plt.plot(np.convolve(rewards, np.ones(1000) / 1000, mode='valid'))
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Performance')
plt.show()
