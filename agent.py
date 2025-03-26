import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from network import policy_net
from torchsummary import summary
import time

class FastLogTable:
    def __init__(self, num_points=1000, x_min=1e-6, x_max=1.0, device='cpu'):
        self.num_points = num_points
        self.x_min = x_min
        self.x_max = x_max
        self.device = device

        self.x_vals = torch.linspace(x_min, x_max, steps=num_points, device=device)
        self.log_vals = torch.log(self.x_vals)
        self.step = (x_max - x_min) / (num_points - 1)

    def __call__(self, x: torch.Tensor):
        #x = torch.clamp(x, min=self.x_min, max=self.x_max)
        #print(x)
        indices = ((x - self.x_min) / self.step ).long()
        #print(indices)
        indices = torch.clamp(indices, 0, self.num_points - 1)
        #print(indices)
        return self.log_vals[indices]

log_table = FastLogTable(num_points=100000,device='cpu')  # встроенный fast log

class WrappedCategorical:
    def __init__(self, probs: torch.Tensor):
        """
        probs: Tensor of shape [num_actions] or [batch_size, num_actions]
        """
        self.probs = probs
        self.device = probs.device
        self.log_p = log_table(probs) 
        self.entropy = self.entropy()
        self.base = torch.distributions.Categorical(probs)

    def sample(self):
        return self.base.sample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        actions: [batch_size] или int
        Возвращает log(p) через fast log
        """
        #if self.probs.dim() == 1:
        #p = self.probs[actions]
        #else:
        #    p = self.probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        #print(f"{p=}")
        action = actions.data[0]

        return self.log_p.data[0][action]

    def entropy(self) -> torch.Tensor:
        """
        Возвращает энтропию: -Σ p * log(p), по строкам (батч-совместимо)
        """
        return -(self.probs * self.log_p).sum(dim=-1)

class PolicyAgent:
    def __init__(self, 
                 input_shape, 
                 num_actions, 
                 device, 
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
        probs = self.policy_net(state_tensor)
        #m = torch.distributions.Categorical(probs)
        #action = m.sample()
        #start = time.time()
        #print(probs)
        #print(m)
        #print(action)
        #print(m.log_prob(action))
        
        # Save entropy for the current action distribution
        #entropy = m.entropy().unsqueeze(0)
        #print(entropy)
        #print(f'time:{time.time()-start}')

        #print("====")
        m = WrappedCategorical(probs)
        action = m.sample()
        self.current_log_probs.append(m.log_prob(action))
        #start = time.time()
        #print(f"{probs=}")
        #print(f"{m=}")
        #print(f"{action=}")
        #print(m.log_prob(action))
        
        # Save entropy for the current action distribution
        entropy = m.entropy.unsqueeze(0)
        #print(entropy)
        #print(f'time:{time.time()-start}')
        #exit()
        #start = time.time()
       # probs = probs.numpy().flatten()
       # print(probs)
       # index = np.random.choice(len(probs), p=probs)
       # log_probs = self.logTable(probs)
       # print(log_probs[index]) 
       # entropy = - (log_probs*probs).sum()
       # print(entropy)
       # print(f'time:{time.time()-start}')
       # exit()
        self.entropy_history.append(entropy)
        return action.item()
    #return int(action.item())

    def select_action(self, state):
        return self._network(state) 
        # Alternative: return self._egreedy_policy(state)

    def store_reward(self, reward):
        self.current_rewards.append(reward)
    
    def finish_episode(self):
        # Store current episode data and reset for next episode
        self.episode_buffer.append((self.current_log_probs, self.current_rewards))
        self.current_log_probs = []
        self.current_rewards = []
        
        # If we've accumulated enough episodes, perform a batch update
        if len(self.episode_buffer) >= self.update_interval:
            all_log_probs = []
            all_returns = []
            
            # Calculate returns for each episode
            for ep_log_probs, ep_rewards in self.episode_buffer:
                R = 0
                ep_returns = []
                
                # Calculate discounted returns
                for r in reversed(ep_rewards):
                    R = r + self.gamma * R
                    ep_returns.insert(0, R)
                
                # Normalize returns
                ep_returns = torch.tensor(ep_returns, dtype=torch.float32).to(self.device)
                ep_returns = (ep_returns - ep_returns.mean()) / (ep_returns.std() + 1e-5)
                
                all_log_probs.extend(ep_log_probs)
                all_returns.append(ep_returns)
            
            # Concatenate returns from all episodes
            all_returns = torch.cat(all_returns)
            
            # Calculate policy loss
            policy_loss = []
            for log_prob, R in zip(all_log_probs, all_returns):
                policy_loss.append(-log_prob * R)
            
            # Add entropy term for exploration
            if self.entropy_history:
                entropy_term = sum(self.entropy_history)
            else:
                entropy_term = 0
            
            # Combine policy loss with entropy bonus
            loss = torch.stack(policy_loss).sum() - self.beta * entropy_term
            
            #print(f"Loss: {loss.item()}")
            
            # Perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Clear buffers
            self.episode_buffer = []
            self.entropy_history = []
