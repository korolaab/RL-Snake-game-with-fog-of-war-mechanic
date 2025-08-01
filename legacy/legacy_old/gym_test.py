import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

# Гиперпараметры
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
num_episodes = 500

class Agent:
    def __init__(self, state_size, action_size, lr):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, self.action_size),
            nn.Softmax(dim=-1)
        )
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.model(state)
        action = torch.multinomial(action_probs, 1).item()
        return action, action_probs
    
    def update_policy(self, states, actions, rewards):
        max_reward_idx = torch.argmax(torch.tensor(rewards))  # Берем индекс лучшего действия
        target_actions = torch.tensor([actions[max_reward_idx]])  # Целевое действие
        
        states = torch.stack(states)
        outputs = self.model(states)  # Получаем распределение вероятностей действий
        
        loss = self.loss_fn(outputs, target_actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Создаем окружение
env = gym.make("CartPole-v1")  # Предположим, что новая среда заменяется здесь
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = 3  # Обновлено для среды с 3 действиями
agent = Agent(STATE_SIZE, ACTION_SIZE, LEARNING_RATE)

# Цикл обучения
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    states = []
    actions = []
    rewards = []
    
    while not done:
        action, action_probs = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        states.append(torch.FloatTensor(state))
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
        total_reward += reward
    
    agent.update_policy(states, actions, rewards)
    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()

