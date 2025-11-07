import requests
import time
import json
import argparse
import sys
import os

from datetime import datetime
import logging

import posix_ipc
import mmap
import struct

import torch
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict
import numpy as np


class SnakeNet(nn.Module):
    """Нейронная сеть для змейки."""
    
    def __init__(self, input_size, hidden_units_1=14, hidden_units_2=12, dropout_rate=0.3):
        super(SnakeNet, self).__init__()
        self.input_size = input_size
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        # HYPEROPT-OPTIMIZED ARCHITECTURE: Best configuration from hyperopt tuning
        # {"hidden_units_1": 14, "activation_1": "Tanh", "hidden_units_2": 8, "activation_2": "Tanh", "dropout_rate": 0.6}
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_units_1),
           # nn.LayerNorm(hidden_units_1),
            nn.Tanh(),           
           # nn.Dropout(dropout_rate),
            nn.Linear(hidden_units_1, hidden_units_2),
           # nn.LayerNorm(hidden_units_2),
            nn.Tanh(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_units_2, hidden_units_2),
           # nn.LayerNorm(hidden_units_2),
            nn.Tanh(),
            nn.Linear(hidden_units_2, 3),
            #nn.LayerNorm(3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)
    
def train():
    # Extract episodes
    episodes = [replay_buffer]#TODO: several episodes
    all_states, all_actions, all_returns = [], [], []

    # Process each episode
    for episode in episodes:
        if not episode:
            continue
        
        # Extract (state, action, reward) from each experience
        states = []
        actions = []
        rewards = []
        
        for exp in episode[1:]:
            # Process state

            states.append(exp[0])
            
            # Convert action name to index
            action_idx = exp[1]
            actions.append(action_idx)
            
            # Store reward
            rewards.append(exp[2])
        
        # Calculate discounted returns (backward through episode)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + args.gamma * G
            returns.insert(0, G)
        
        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32)
        
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Add to batch
        all_states.extend(states)
        all_actions.extend(actions)
        all_returns.extend(returns.tolist())

    # Skip if no data
    if not all_states:
        return False

    # Create tensors
    states_tensor = torch.stack(all_states)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)
    returns_tensor = torch.tensor(all_returns, dtype=torch.float32)

    # Forward pass and loss
    model.train()
    action_probs = model(states_tensor)
    m = torch.distributions.Categorical(action_probs)
    log_probs = m.log_prob(actions_tensor)
    entropy = m.entropy()
    # REINFORCE loss
    ent = entropy.mean()
    loss1 = -(log_probs * returns_tensor).mean()
    loss =  loss1- args.beta * ent
    optimizer.zero_grad()                
    loss.backward()  
    
    # Calculate gradient norm
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    optimizer.step()
    
    # Calculate additional metrics
    entropy_std = entropy.std().item() if len(entropy) > 1 else 0.0
    returns_mean = returns_tensor.mean().item()
    returns_std = returns_tensor.std().item()
    
    # Action distribution analysis
    action_counts = torch.bincount(actions_tensor, minlength=3)
    action_freqs = action_counts.float() / len(actions_tensor)
    
    return {
        'loss': loss.item(),
        'policy_loss': loss1.item(),
        'entropy_mean': entropy.mean().item(),
        'entropy_std': entropy_std,
        'grad_norm': grad_norm.item(),
        'returns_mean': returns_mean,
        'returns_std': returns_std,
        'action_0_freq': action_freqs[0].item(),
        'action_1_freq': action_freqs[1].item(),
        'action_2_freq': action_freqs[2].item()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural network agent local training only (no gRPC)")
    parser.add_argument("--snake-id", type=int, required=True, help="Snake ID for this agent")
    parser.add_argument("--log-file", type=str, default="agent_log.json", help="Log file path")
    parser.add_argument("--env-host", type=str, default="localhost:5000", help="Environment host URL")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=5, help="Episodes per batch")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (gamma) for RL")
    parser.add_argument("--beta", type=float, default=0.1, help="Entropy bonus (beta)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Number of episodes before exit (overrides env N_EPISODES)")

    args = parser.parse_args()
  
    # открываем shared memory (создано Clock)
    while True:
        try:
            shm = posix_ipc.SharedMemory("/game_state")
            break
        except posix_ipc.ExistentialError:
            print("[ENV] waiting for /game_state shm...")
            time.sleep(0.1)
    
    # открываем семафоры
    while True:
        try:
            sem_inf_tick = posix_ipc.Semaphore("/sem_inf_tick")
            break
        except posix_ipc.ExistentialError:
            print("[INF] waiting for /sem_env_tick semaphore...")
            time.sleep(0.1)
 
    while True:
        try:
            sem_inf_done = posix_ipc.Semaphore("/sem_inf_done")
            break
        except posix_ipc.ExistentialError:
            print("[INF] waiting for /sem_inf_done semaphore...")
            time.sleep(0.1)
    
    while True:
        try:
            shm_ctrl = posix_ipc.SharedMemory("/inf_control")   
            break
        except posix_ipc.ExistentialError:
            print("[ENV] waiting for /inf_control shm...")
            time.sleep(0.1)


    header_fmt = "<d?q"
    header_size = struct.calcsize(header_fmt)
    # Calculate vision size dynamically (consistent with environment)
    vision_radius = 5  # Should match environment configuration
    vision_size = 2 * vision_radius * (vision_radius + 1) + 2
    total_size = header_size + (vision_size * 8) * 2

    mapfile_ctrl = mmap.mmap(shm_ctrl.fd, shm_ctrl.size)
    ctrl_fmt = "=idddddddddd"  # do_train, loss, policy_loss, entropy_mean, entropy_std, grad_norm, returns_mean, returns_std, action_0_freq, action_1_freq, action_2_freq

    mapfile = mmap.mmap(shm.fd, total_size)

    
    model = SnakeNet( input_size = vision_size * 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #print("[INF] Created Model")

    # Store as list of tuples
    replay_buffer = []

    # Episode counter for model checkpointing
    episode_count = 0
    
    prev_vision_tensor = torch.zeros(vision_size*2)
    prev_action = 0
    while True:

        sem_inf_tick.acquire()
       # print("[INF] got signal")
        
        # Check if this is a do_train request
        do_train, loss, policy_loss, entropy_mean, entropy_std, grad_norm, returns_mean, returns_std, action_0_freq, action_1_freq, action_2_freq = struct.unpack_from(ctrl_fmt, mapfile_ctrl, 0)
        
        if do_train == 1:  
            #print("[ENV] Train flag recieved")
            episode_count += 1

            # Read reward and game state from shared memory
            reward, game_over, action = struct.unpack_from(header_fmt, mapfile, 0)

            # читаем vision как np.float64 (consistent with inference section)
            vision = np.frombuffer(mapfile, dtype=np.float64,
                        count=vision_size*2, offset=header_size)
            vision_tensor = torch.from_numpy(vision.astype(np.float32))

            # Each experience: (state, action, reward, next_state, done) 
            experience = (
                prev_vision_tensor,      # torch.Tensor
                prev_action,     # torch.Tensor or int
                reward,            # float
                vision_tensor
            )


            metrics = train()
            
            # Save model checkpoint every 50 episodes
            if episode_count % 50 == 0:
                checkpoint_path = f"/logs/model_checkpoint_episode_{episode_count}.pth"
                torch.save({
                    'epoch': episode_count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': metrics['loss'],
                    'architecture': {
                        'input_size': model.input_size,
                        'hidden_units_1': model.hidden_units_1,
                        'hidden_units_2': model.hidden_units_2
                    },
                    'hyperparameters': {
                        'learning_rate': args.learning_rate,
                        'gamma': args.gamma,
                        'beta': args.beta
                    }
                }, checkpoint_path)
                print(f"[INF] Saved model checkpoint: {checkpoint_path}")
            
            struct.pack_into(ctrl_fmt, mapfile_ctrl, 0,
                             0,  # do_train flag
                             metrics['loss'],
                             metrics['policy_loss'],
                             metrics['entropy_mean'],
                             metrics['entropy_std'],
                             metrics['grad_norm'],
                             metrics['returns_mean'],
                             metrics['returns_std'],
                             metrics['action_0_freq'],
                             metrics['action_1_freq'],
                             metrics['action_2_freq'])
                             
            replay_buffer = []
            prev_action = 0
            prev_vision_tensor = torch.zeros(vision_size*2)
        else:
            reward, game_over, action = struct.unpack_from(header_fmt, mapfile, 0)

            # читаем vision как np.int8
            vision = np.frombuffer(mapfile, dtype=np.float64,
                        count=vision_size*2, offset=header_size)
            vision_tensor = torch.from_numpy(vision.astype(np.float32))


            action_offset = struct.calcsize("<d?") 
           
            
            with torch.no_grad():
                #model.eval()
                action_probs = model(vision_tensor)
                m = torch.distributions.Categorical(action_probs)
                action = m.sample()

            # Each experience: (state, action, reward, next_state, done) 
            experience = (
                prev_vision_tensor,      # torch.Tensor
                prev_action,     # torch.Tensor or int
                reward,            # float
                vision_tensor
            )
            prev_vision_tensor = vision_tensor.clone()
            prev_action = action
            replay_buffer.append(experience)

            
            struct.pack_into("q", mapfile, action_offset, action)
           # print(f"[INF] wrote action {action}")
        # сигналим Clock, что данные готовы
        sem_inf_done.release()

    shm.close_fd()