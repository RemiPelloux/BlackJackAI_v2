import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from blackjack_env import BlackjackEnv


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.env.action_space.n)
        )
        return model.cuda()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).cuda().unsqueeze(0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.env.action_space.n)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).cuda().unsqueeze(0)
            next_state = torch.FloatTensor(next_state).cuda().unsqueeze(0)
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.target_model(next_state)[0]).item())
            target_f = self.model(state).clone()
            target_f[0][action] = target
            self.model.zero_grad()
            loss = nn.MSELoss()(self.model(state)[0], target_f[0])
            loss.backward()
            for param in self.model.parameters():
                param.data -= self.learning_rate * param.grad.data
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.update_target_model()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
