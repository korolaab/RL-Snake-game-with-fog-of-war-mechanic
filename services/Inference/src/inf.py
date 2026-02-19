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


class STEBinarize(torch.autograd.Function):
    """Straight-Through Estimator for binary activation: round in forward, pass gradient through sigmoid in backward."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SnakeNet(nn.Module):
    """Neural network for snake with optional talk vector communication."""

    def __init__(self, input_size, talk_size=0, hidden_units_1=14, hidden_units_2=12, dropout_rate=0.3, comm_dropout=0.0):
        super(SnakeNet, self).__init__()
        self.input_size = input_size
        self.talk_size = talk_size
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2

        total_input = input_size + talk_size

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Linear(total_input, hidden_units_1),
            nn.Tanh(),
        )
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_units_1, hidden_units_2),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        # Policy head (layers 3-4)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_units_2, hidden_units_2),
            nn.Tanh(),
            nn.Linear(hidden_units_2, 3),
            nn.Softmax(dim=-1)
        )
        # Value head (branches from layer2 output)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_units_2, hidden_units_2),
            nn.Tanh(),
            nn.Linear(hidden_units_2, 1)
        )
        # Talk head (branches from layer2 output)
        if talk_size > 0:
            self.talk_head = nn.Sequential(
                nn.Linear(hidden_units_2, talk_size),
                nn.Sigmoid(),
            )
            self.comm_dropout = nn.Dropout(comm_dropout)

    def forward(self, x, talk_in=None):
        if self.talk_size > 0 and talk_in is not None:
            talk_in = self.comm_dropout(talk_in)
            x = torch.cat([x, talk_in], dim=-1)
        elif self.talk_size > 0:
            x = torch.cat([x, torch.zeros(x.shape[:-1] + (self.talk_size,))], dim=-1)

        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        action_probs = self.policy_head(h2)
        value = self.value_head(h2).squeeze(-1)

        if self.talk_size > 0:
            talk_out = self.talk_head(h2)
            talk_out = STEBinarize.apply(talk_out)
            return action_probs, talk_out, value

        return action_probs, None, value


def train():
    episodes = [replay_buffer]
    all_states1, all_states2, all_actions1, all_actions2, all_returns = [], [], [], [], []
    all_old_log_probs1, all_old_log_probs2 = [], []
    all_talk_in1, all_talk_in2 = [], []

    for episode in episodes:
        if not episode:
            continue

        rewards = []
        for exp in episode[1:]:
            if num_snakes == 2:
                s1, s2, a1, a2, old_lp1, old_lp2, ti1, ti2, r = exp
                all_states1.append(s1)
                all_states2.append(s2)
                all_actions1.append(a1)
                all_actions2.append(a2)
                all_old_log_probs1.append(old_lp1)
                all_old_log_probs2.append(old_lp2)
                if ti1 is not None:
                    all_talk_in1.append(ti1)
                    all_talk_in2.append(ti2)
            else:
                s, a, old_lp, r = exp[0], exp[1], exp[2], exp[3]
                all_states1.append(s)
                all_actions1.append(a)
                all_old_log_probs1.append(old_lp)
            rewards.append(r)

        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + args.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        all_returns.extend(returns.tolist())

    if not all_states1:
        return False

    states1_tensor = torch.stack(all_states1)
    actions1_tensor = torch.tensor(all_actions1, dtype=torch.long)
    returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
    old_log_probs1_tensor = torch.tensor(all_old_log_probs1, dtype=torch.float32)

    model.train()

    if num_snakes == 2:
        states2_tensor = torch.stack(all_states2)
        actions2_tensor = torch.tensor(all_actions2, dtype=torch.long)
        old_log_probs2_tensor = torch.tensor(all_old_log_probs2, dtype=torch.float32)

        talk_in1_tensor = torch.stack(all_talk_in1) if all_talk_in1 else None
        talk_in2_tensor = torch.stack(all_talk_in2) if all_talk_in2 else None

        for _ in range(args.ppo_epochs):
            # Forward pass with saved talk_in from experience
            probs1, _, values1 = model(states1_tensor, talk_in1_tensor)
            probs2, _, values2 = model(states2_tensor, talk_in2_tensor)

            m1 = torch.distributions.Categorical(probs1)
            m2 = torch.distributions.Categorical(probs2)
            new_log_probs1 = m1.log_prob(actions1_tensor)
            new_log_probs2 = m2.log_prob(actions2_tensor)
            entropy1 = m1.entropy()
            entropy2 = m2.entropy()

            # Shared advantage
            values_mean = (values1 + values2) / 2
            advantage = returns_tensor - values_mean.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Separate PPO loss for each snake
            ratio1 = torch.exp(new_log_probs1 - old_log_probs1_tensor)
            surr1_a = ratio1 * advantage
            surr1_b = torch.clamp(ratio1, 1 - args.eps_clip, 1 + args.eps_clip) * advantage
            policy_loss1 = -torch.min(surr1_a, surr1_b).mean()

            ratio2 = torch.exp(new_log_probs2 - old_log_probs2_tensor)
            surr2_a = ratio2 * advantage
            surr2_b = torch.clamp(ratio2, 1 - args.eps_clip, 1 + args.eps_clip) * advantage
            policy_loss2 = -torch.min(surr2_a, surr2_b).mean()

            value_loss1 = nn.functional.mse_loss(values1, returns_tensor)
            value_loss2 = nn.functional.mse_loss(values2, returns_tensor)

            ent = (entropy1.mean() + entropy2.mean()) / 2
            loss = policy_loss1 + policy_loss2 + args.value_coef * (value_loss1 + value_loss2) - args.beta * ent

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            optimizer.step()

        entropy_all = torch.cat([entropy1, entropy2])
        actions_all = torch.cat([actions1_tensor, actions2_tensor])
        policy_loss_total = policy_loss1.item() + policy_loss2.item()
    else:
        for _ in range(args.ppo_epochs):
            probs1, _, values1 = model(states1_tensor, None)
            m1 = torch.distributions.Categorical(probs1)
            new_log_probs1 = m1.log_prob(actions1_tensor)
            entropy1 = m1.entropy()

            adv1 = returns_tensor - values1.detach()
            adv1 = (adv1 - adv1.mean()) / (adv1.std() + 1e-8)

            ratio1 = torch.exp(new_log_probs1 - old_log_probs1_tensor)
            surr1 = ratio1 * adv1
            surr2 = torch.clamp(ratio1, 1 - args.eps_clip, 1 + args.eps_clip) * adv1
            policy_loss1 = -torch.min(surr1, surr2).mean()
            value_loss1 = nn.functional.mse_loss(values1, returns_tensor)

            ent = entropy1.mean()
            loss = policy_loss1 + args.value_coef * value_loss1 - args.beta * ent

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            optimizer.step()

        entropy_all = entropy1
        actions_all = actions1_tensor
        policy_loss_total = policy_loss1.item()

    entropy_std = entropy_all.std().item() if len(entropy_all) > 1 else 0.0
    returns_mean = returns_tensor.mean().item()
    returns_std = returns_tensor.std().item()

    action_counts = torch.bincount(actions_all, minlength=3)
    action_freqs = action_counts.float() / len(actions_all)

    return {
        'loss': loss.item(),
        'policy_loss': policy_loss_total,
        'entropy_mean': ent.item(),
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
    parser.add_argument("--max-episodes", type=int, default=None, help="Number of episodes before exit")
    parser.add_argument("--num-snakes", type=int, default=1, help="Number of snakes (1 or 2)")
    parser.add_argument("--talk-size", type=int, default=12, help="Size of talk vector for inter-snake communication")
    parser.add_argument("--comm-dropout", type=float, default=0.0, help="Dropout rate on talk input (0.0=full comm, 1.0=no comm)")
    parser.add_argument("--hidden1", type=int, default=14, help="Hidden units in layer 1")
    parser.add_argument("--hidden2", type=int, default=12, help="Hidden units in layer 2")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO optimization epochs per episode")
    parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")

    args = parser.parse_args()
    num_snakes = args.num_snakes

    # открываем shared memory (создано Clock)
    while True:
        try:
            shm = posix_ipc.SharedMemory("/game_state")
            break
        except posix_ipc.ExistentialError:
            print("[INF] waiting for /game_state shm...")
            time.sleep(0.1)

    # открываем семафоры
    while True:
        try:
            sem_inf_tick = posix_ipc.Semaphore("/sem_inf_tick")
            break
        except posix_ipc.ExistentialError:
            print("[INF] waiting for /sem_inf_tick semaphore...")
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
            print("[INF] waiting for /inf_control shm...")
            time.sleep(0.1)


    if num_snakes == 2:
        header_fmt = "<d?qq"
    else:
        header_fmt = "<d?q"
    header_size = struct.calcsize(header_fmt)
    vision_radius = 5
    vision_size = 2 * vision_radius * (vision_radius + 1) + 2
    state_bytes = (vision_size * 8) * 2  # one state buffer
    total_size = header_size + state_bytes * num_snakes

    mapfile_ctrl = mmap.mmap(shm_ctrl.fd, shm_ctrl.size)
    ctrl_fmt = "=idddddddddd"

    mapfile = mmap.mmap(shm.fd, total_size)

    obs_size = vision_size * 2
    talk_size = args.talk_size if num_snakes == 2 else 0

    id_size = 2 if num_snakes == 2 else 0
    snake1_id = torch.tensor([1.0, 0.0]) if num_snakes == 2 else None
    snake2_id = torch.tensor([0.0, 1.0]) if num_snakes == 2 else None
    model = SnakeNet(input_size=obs_size + id_size, talk_size=talk_size, hidden_units_1=args.hidden1, hidden_units_2=args.hidden2, comm_dropout=args.comm_dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    replay_buffer = []
    episode_count = 0

    prev_vision1 = torch.zeros(obs_size + id_size)
    prev_vision2 = torch.zeros(obs_size + id_size) if num_snakes == 2 else None
    prev_action1 = 0
    prev_action2 = 0
    prev_old_log_prob1 = 0.0
    prev_old_log_prob2 = 0.0
    prev_talk1 = torch.zeros(talk_size) if talk_size > 0 else None
    prev_talk2 = torch.zeros(talk_size) if talk_size > 0 else None

    while True:
        sem_inf_tick.acquire()

        do_train, loss, policy_loss, entropy_mean, entropy_std, grad_norm, returns_mean, returns_std, action_0_freq, action_1_freq, action_2_freq = struct.unpack_from(ctrl_fmt, mapfile_ctrl, 0)

        if do_train == 1:
            episode_count += 1

            # Read final state for last experience
            header_data = struct.unpack_from(header_fmt, mapfile, 0)
            reward = header_data[0]

            vision1 = np.frombuffer(mapfile, dtype=np.float64, count=vision_size*2, offset=header_size)
            v1_tensor = torch.from_numpy(vision1.astype(np.float32))

            if num_snakes == 2:
                vision2 = np.frombuffer(mapfile, dtype=np.float64, count=vision_size*2, offset=header_size + state_bytes)
                v2_tensor = torch.from_numpy(vision2.astype(np.float32))

            metrics = train()

            if episode_count % 50 == 0:
                checkpoint_path = f"/logs/model_checkpoint_episode_{episode_count}.pth"
                torch.save({
                    'epoch': episode_count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': metrics['loss'],
                    'architecture': {
                        'input_size': model.input_size,
                        'talk_size': model.talk_size,
                        'hidden_units_1': model.hidden_units_1,
                        'hidden_units_2': model.hidden_units_2
                    },
                    'hyperparameters': {
                        'learning_rate': args.learning_rate,
                        'gamma': args.gamma,
                        'beta': args.beta,
                        'talk_size': talk_size,
                        'comm_dropout': args.comm_dropout,
                        'num_snakes': num_snakes,
                        'ppo_epochs': args.ppo_epochs,
                        'eps_clip': args.eps_clip,
                        'value_coef': args.value_coef
                    }
                }, checkpoint_path)
                print(f"[INF] Saved model checkpoint: {checkpoint_path}")

            struct.pack_into(ctrl_fmt, mapfile_ctrl, 0,
                             0,
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
            prev_action1 = 0
            prev_action2 = 0
            prev_old_log_prob1 = 0.0
            prev_old_log_prob2 = 0.0
            prev_vision1 = torch.zeros(obs_size + id_size)
            if num_snakes == 2:
                prev_vision2 = torch.zeros(obs_size + id_size)
            if talk_size > 0:
                prev_talk1 = torch.zeros(talk_size)
                prev_talk2 = torch.zeros(talk_size)
        else:
            header_data = struct.unpack_from(header_fmt, mapfile, 0)
            reward = header_data[0]

            vision1 = np.frombuffer(mapfile, dtype=np.float64, count=vision_size*2, offset=header_size)
            v1_tensor = torch.from_numpy(vision1.astype(np.float32))

            if num_snakes == 2:
                vision2 = np.frombuffer(mapfile, dtype=np.float64, count=vision_size*2, offset=header_size + state_bytes)
                v2_tensor = torch.from_numpy(vision2.astype(np.float32))

                with torch.no_grad():
                    v1_with_id = torch.cat([v1_tensor, snake1_id])
                    v2_with_id = torch.cat([v2_tensor, snake2_id])

                    # Bidirectional: snake1 receives prev_talk2, snake2 receives prev_talk1
                    probs1, talk_out1, _ = model(v1_with_id, prev_talk2)
                    probs2, talk_out2, _ = model(v2_with_id, prev_talk1)

                    m1 = torch.distributions.Categorical(probs1)
                    action1 = m1.sample()
                    old_log_prob1 = m1.log_prob(action1)

                    m2 = torch.distributions.Categorical(probs2)
                    action2 = m2.sample()
                    old_log_prob2 = m2.log_prob(action2)

                # Save talk_in used this step (for training)
                talk_in1_saved = prev_talk2.clone() if prev_talk2 is not None else None
                talk_in2_saved = prev_talk1.clone() if prev_talk1 is not None else None

                experience = (
                    prev_vision1,
                    prev_vision2,
                    prev_action1,
                    prev_action2,
                    prev_old_log_prob1,
                    prev_old_log_prob2,
                    talk_in1_saved,
                    talk_in2_saved,
                    reward,
                )
                prev_vision1 = v1_with_id.clone()
                prev_vision2 = v2_with_id.clone()
                prev_action1 = action1
                prev_action2 = action2
                prev_old_log_prob1 = old_log_prob1.item()
                prev_old_log_prob2 = old_log_prob2.item()
                if talk_out1 is not None:
                    prev_talk1 = talk_out1.clone()
                    prev_talk2 = talk_out2.clone()
                replay_buffer.append(experience)

                # Write both actions
                action_offset = struct.calcsize("<d?")
                struct.pack_into("q", mapfile, action_offset, action1)
                struct.pack_into("q", mapfile, action_offset + 8, action2)
            else:
                with torch.no_grad():
                    probs1, _, _ = model(v1_tensor, None)
                    m1 = torch.distributions.Categorical(probs1)
                    action1 = m1.sample()
                    old_log_prob1 = m1.log_prob(action1)

                experience = (
                    prev_vision1,
                    prev_action1,
                    prev_old_log_prob1,
                    reward,
                )
                prev_vision1 = v1_tensor.clone()
                prev_action1 = action1
                prev_old_log_prob1 = old_log_prob1.item()
                replay_buffer.append(experience)

                action_offset = struct.calcsize("<d?")
                struct.pack_into("q", mapfile, action_offset, action1)

        sem_inf_done.release()

    shm.close_fd()
