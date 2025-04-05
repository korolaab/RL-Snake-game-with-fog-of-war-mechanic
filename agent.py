import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from network import policy_net
from torchsummary import summary

class PolicyAgent:
    def __init__(self, 
                 input_shape, 
                 num_actions, 
                 device, 
                 agent_id,
                 epsilon=0.2, 
                 lr=1e-3,
                 beta=0.1, 
                 gamma=0.99, 
                 update_interval=1,
                 params = {"hidden_units_1": 8,
                           "activation_1" : "Tanh",
                           "hidden_units_2": 16,
                           "activation_2": "Tanh",
                           "dropout_rate": 0.5}):
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta  # Entropy regularization coefficient
        self.n_actions = num_actions
        self.id = agent_id

        self.states_history = [] 
        self.actions_history = []

        # Initialize neural network
        #self.policy_net = PolicyNet(input_shape, num_actions).to(device)
        self.policy_net = policy_net(input_shape = input_shape, 
                                     num_actions = num_actions, 
                                     hidden_units_1 = params['hidden_units_1'],
                                     activation_1 = params['activation_1'],
                                     hidden_units_2 = params['hidden_units_2'],
                                     activation_2 = params['activation_2'],
                                     dropout_rate = params['dropout_rate']).to(device)
        summary(self.policy_net, input_size=(1, input_shape[0], input_shape[1]), device=device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Data storage for one episode
        self.current_log_probs = []
        self.current_rewards = []
        
        # Buffer for accumulating data over several episodes
        self.entropy_history = []
        self.episode_buffer = []
        self.update_interval = update_interval

    def _egreedy_policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions) 
        else:
            return self._network(state)

    def _network(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy_net(state_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        self.actions_history.append(action)
        self.states_history.append(state)
        
        return int(action.item())

    def select_action(self, state):
        return self._network(state) 
        # Alternative: return self._egreedy_policy(state)

    def store_reward(self, reward):
        self.current_rewards.append(reward)
    
    def finish_episode(self):
        states_tensor = torch.tensor(self.states_history, dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.actions_history)
        probs = self.policy_net(states_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_probs = m.log_prob(actions)
        entropy = m.entropy()
        #self.entropy_history.append(entropy)

        # Store current episode data and reset for next episode
        #self.current_log_probs = []
        #self.current_rewards = []
        # Calculate returns for each episode
        all_returns = []

        R = 0
        ep_returns = []
        
        # Calculate discounted returns
        for r in reversed(self.current_rewards):
            R = r + self.gamma * R
            ep_returns.insert(0, R)
        
        # Normalize returns
        ep_returns = torch.tensor(ep_returns, dtype=torch.float32).to(self.device)
        ep_returns = (ep_returns - ep_returns.mean()) / (ep_returns.std() + 1e-5)
        
        
        # Concatenate returns from all episodes
        
        # Calculate policy loss
        #policy_loss = []
        #for log_prob, R in zip(all_log_probs, all_returns):
        #    policy_loss.append(-log_prob * R)
        
        # Add entropy term for exploration
        #if self.entropy_history:
        #    entropy_term = torch.cat(self.entropy_history).sum()
        #else:
        #    entropy_term = 0
        
        # Combine policy loss with entropy bonus
        loss = -(log_probs * ep_returns).sum() - self.beta * entropy.sum()
        
        #print(f"Loss: {loss.item()}")

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.episode_buffer = []
        self.entropy_history = []
        self.states_history = []
        self.actions_history = []
        self.current_rewards = []
