# test_dqn_agent.py
import torch
import numpy as np
from blackjack_env import BlackjackEnv
from dqn_blackjack import DQN

env = BlackjackEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Load the trained model
model = DQN(state_size, action_size).cuda()
model.load_state_dict(torch.load("dqn_blackjack.pth"))
model.eval()


def act(state):
    state = torch.FloatTensor(state).unsqueeze(0).cuda()
    with torch.no_grad():
        act_values = model(state)
    return torch.argmax(act_values[0]).item()


# Test the agent
num_episodes = 1000
total_rewards = 0

for episode in range(num_episodes):
    state = env.reset()
    state = np.array(state).reshape(1, -1)  # Ensure state is a 2D array
    done = False
    episode_reward = 0

    while not done:
        action = act(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        next_state = np.array(next_state).reshape(1, -1)  # Ensure next_state is a 2D array
        state = next_state

    total_rewards += episode_reward

average_reward = total_rewards / num_episodes
print(f"Average reward over {num_episodes} episodes: {average_reward}")
