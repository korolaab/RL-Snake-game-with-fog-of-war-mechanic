#!/usr/bin/env python3
"""Single-script training loop for Snake RL. No Docker, no IPC."""

import argparse
import json
import os
import sys

import mlflow
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services', 'Env', 'src'))
from game import SnakeGame


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class STEBinarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SnakeNet(nn.Module):
    def __init__(self, input_size, talk_size=0, hidden_units_1=128, hidden_units_2=64):
        super().__init__()
        self.input_size = input_size
        self.talk_size = talk_size
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2

        total_input = input_size + talk_size

        self.layer1 = nn.Sequential(nn.Linear(total_input, hidden_units_1), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(hidden_units_1, hidden_units_2), nn.Tanh())
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_units_2, hidden_units_2), nn.Tanh(),
            nn.Linear(hidden_units_2, 3), nn.Softmax(dim=-1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_units_2, hidden_units_2), nn.Tanh(),
            nn.Linear(hidden_units_2, 1)
        )
        if talk_size > 0:
            self.talk_head = nn.Sequential(nn.Linear(hidden_units_2, talk_size), nn.Sigmoid())

    def forward(self, x, talk_in=None):
        if self.talk_size > 0:
            if talk_in is not None:
                x = torch.cat([x, talk_in], dim=-1)
            else:
                x = torch.cat([x, torch.zeros(x.shape[:-1] + (self.talk_size,))], dim=-1)

        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        action_probs = self.policy_head(h2)
        value = self.value_head(h2).squeeze(-1)

        if self.talk_size > 0:
            talk_out = STEBinarize.apply(self.talk_head(h2))
            return action_probs, talk_out, value

        return action_probs, None, value


# --------------------------------------------------------------------------- #
# Episode collection
# --------------------------------------------------------------------------- #

def state_to_tensor(state_matrix):
    return torch.tensor(state_matrix.flatten(), dtype=torch.float32)


def collect_episode(env, model, args):
    """Run one episode; return (replay_buffer, stats)."""
    env.reset()
    num_snakes = args.num_snakes
    talk_size = args.talk_size if num_snakes == 2 else 0

    # snake id one-hot for dual mode
    snake1_id = torch.tensor([1.0, 0.0]) if num_snakes == 2 else None
    snake2_id = torch.tensor([0.0, 1.0]) if num_snakes == 2 else None

    obs_size = (2 * args.vision_radius * (args.vision_radius + 1) + 2) * 2
    id_size = 2 if num_snakes == 2 else 0
    full_obs_size = obs_size + id_size

    prev_v1 = torch.zeros(full_obs_size)
    prev_v2 = torch.zeros(full_obs_size) if num_snakes == 2 else None
    prev_a1, prev_a2 = 0, 0
    prev_lp1, prev_lp2 = 0.0, 0.0
    prev_talk1 = torch.zeros(talk_size) if talk_size > 0 else None
    prev_talk2 = torch.zeros(talk_size) if talk_size > 0 else None

    replay_buffer = []
    total_reward = 0.0
    done = False

    # Get initial state
    if num_snakes == 2:
        state = env.get_state()
        s1, s2 = state
    else:
        s1 = env.get_state()

    while not done:
        v1 = state_to_tensor(s1)
        if num_snakes == 2:
            v2 = state_to_tensor(s2)
            v1_in = torch.cat([v1, snake1_id])
            v2_in = torch.cat([v2, snake2_id])

            with torch.no_grad():
                probs1, talk_out1, _ = model(v1_in, prev_talk2)
                probs2, talk_out2, _ = model(v2_in, prev_talk1)

            m1 = torch.distributions.Categorical(probs1)
            a1 = m1.sample()
            lp1 = m1.log_prob(a1)

            m2 = torch.distributions.Categorical(probs2)
            a2 = m2.sample()
            lp2 = m2.log_prob(a2)

            talk_in1_saved = prev_talk2.clone() if prev_talk2 is not None else None
            talk_in2_saved = prev_talk1.clone() if prev_talk1 is not None else None

            experience = (prev_v1, prev_v2, prev_a1, prev_a2, prev_lp1, prev_lp2,
                          talk_in1_saved, talk_in2_saved, None)  # reward filled below

            (state_next, reward, done) = env.update(a1.item(), a2.item())
            if not done:
                s1, s2 = state_next
            total_reward += reward

            # Store with reward
            replay_buffer.append((*experience[:-1], reward))

            prev_v1 = v1_in.clone()
            prev_v2 = v2_in.clone()
            prev_a1, prev_a2 = a1, a2
            prev_lp1, prev_lp2 = lp1.item(), lp2.item()
            if talk_out1 is not None:
                prev_talk1 = talk_out1.clone()
                prev_talk2 = talk_out2.clone()
        else:
            v1_in = v1
            with torch.no_grad():
                probs1, _, _ = model(v1_in, None)
            m1 = torch.distributions.Categorical(probs1)
            a1 = m1.sample()
            lp1 = m1.log_prob(a1)

            experience = (prev_v1, prev_a1, prev_lp1, None)

            (s1_next, reward, done) = env.update(a1.item())
            if not done:
                s1 = s1_next
            total_reward += reward

            replay_buffer.append((prev_v1, prev_a1, prev_lp1, reward))

            prev_v1 = v1_in.clone()
            prev_a1 = a1
            prev_lp1 = lp1.item()

    stats = {
        'length': env.steps if hasattr(env, 'steps') else len(replay_buffer),
        'apples': env.eaten_apples,
        'total_reward': total_reward,
        'steps': len(replay_buffer),
    }
    return replay_buffer, stats


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_on_episode(replay_buffer, model, optimizer, args):
    """PPO + GAE update on one episode's replay buffer."""
    num_snakes = args.num_snakes
    if not replay_buffer:
        return None

    if num_snakes == 2:
        all_s1, all_s2, all_a1, all_a2 = [], [], [], []
        all_lp1, all_lp2 = [], []
        all_ti1, all_ti2 = [], []
        rewards = []
        for exp in replay_buffer:
            s1, s2, a1, a2, lp1, lp2, ti1, ti2, r = exp
            all_s1.append(s1); all_s2.append(s2)
            all_a1.append(a1); all_a2.append(a2)
            all_lp1.append(lp1); all_lp2.append(lp2)
            if ti1 is not None:
                all_ti1.append(ti1); all_ti2.append(ti2)
            rewards.append(r)

        s1_t = torch.stack(all_s1)
        s2_t = torch.stack(all_s2)
        a1_t = torch.stack(all_a1) if isinstance(all_a1[0], torch.Tensor) else torch.tensor(all_a1, dtype=torch.long)
        a2_t = torch.stack(all_a2) if isinstance(all_a2[0], torch.Tensor) else torch.tensor(all_a2, dtype=torch.long)
        lp1_t = torch.tensor(all_lp1, dtype=torch.float32)
        lp2_t = torch.tensor(all_lp2, dtype=torch.float32)
        r_t = torch.tensor(rewards, dtype=torch.float32)
        ti1_t = torch.stack(all_ti1) if all_ti1 else None
        ti2_t = torch.stack(all_ti2) if all_ti2 else None

        with torch.no_grad():
            _, _, v1_old = model(s1_t, ti1_t)
            _, _, v2_old = model(s2_t, ti2_t)
            v_old = (v1_old + v2_old) / 2

        T = len(r_t)
        adv = torch.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            nv = v_old[t + 1].item() if t + 1 < T else 0.0
            delta = r_t[t] + args.gamma * nv - v_old[t]
            gae = delta + args.gamma * args.gae_lambda * gae
            adv[t] = gae
        ret = adv + v_old
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        model.train()
        for _ in range(args.ppo_epochs):
            p1, _, v1 = model(s1_t, ti1_t)
            p2, _, v2 = model(s2_t, ti2_t)
            m1 = torch.distributions.Categorical(p1)
            m2 = torch.distributions.Categorical(p2)
            nlp1 = m1.log_prob(a1_t)
            nlp2 = m2.log_prob(a2_t)
            ent1 = m1.entropy(); ent2 = m2.entropy()

            r1 = torch.exp(nlp1 - lp1_t)
            r2 = torch.exp(nlp2 - lp2_t)
            pl1 = -torch.min(r1 * adv, torch.clamp(r1, 1 - args.eps_clip, 1 + args.eps_clip) * adv).mean()
            pl2 = -torch.min(r2 * adv, torch.clamp(r2, 1 - args.eps_clip, 1 + args.eps_clip) * adv).mean()
            vl1 = nn.functional.mse_loss(v1, ret)
            vl2 = nn.functional.mse_loss(v2, ret)
            ent = (ent1.mean() + ent2.mean()) / 2
            loss = pl1 + pl2 + args.value_coef * (vl1 + vl2) - args.beta * ent
            optimizer.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            optimizer.step()

        actions_all = torch.cat([a1_t, a2_t])
        return {
            'loss': loss.item(),
            'policy_loss': pl1.item() + pl2.item(),
            'value_loss': vl1.item() + vl2.item(),
            'entropy_mean': ent.item(),
            'grad_norm': gn.item(),
            'returns_mean': ret.mean().item(),
            'returns_std': ret.std().item(),
            'action_0_freq': (actions_all == 0).float().mean().item(),
            'action_1_freq': (actions_all == 1).float().mean().item(),
            'action_2_freq': (actions_all == 2).float().mean().item(),
        }
    else:
        all_s, all_a, all_lp, rewards = [], [], [], []
        for exp in replay_buffer:
            s, a, lp, r = exp
            all_s.append(s); all_a.append(a); all_lp.append(lp); rewards.append(r)

        s_t = torch.stack(all_s)
        a_t = torch.stack(all_a) if isinstance(all_a[0], torch.Tensor) else torch.tensor(all_a, dtype=torch.long)
        lp_t = torch.tensor(all_lp, dtype=torch.float32)
        r_t = torch.tensor(rewards, dtype=torch.float32)

        with torch.no_grad():
            _, _, v_old = model(s_t, None)

        T = len(r_t)
        adv = torch.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            nv = v_old[t + 1].item() if t + 1 < T else 0.0
            delta = r_t[t] + args.gamma * nv - v_old[t]
            gae = delta + args.gamma * args.gae_lambda * gae
            adv[t] = gae
        ret = adv + v_old
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        model.train()
        for _ in range(args.ppo_epochs):
            p1, _, v1 = model(s_t, None)
            m1 = torch.distributions.Categorical(p1)
            nlp1 = m1.log_prob(a_t)
            ent1 = m1.entropy()
            r1 = torch.exp(nlp1 - lp_t)
            pl1 = -torch.min(r1 * adv, torch.clamp(r1, 1 - args.eps_clip, 1 + args.eps_clip) * adv).mean()
            vl1 = nn.functional.mse_loss(v1, ret)
            ent = ent1.mean()
            loss = pl1 + args.value_coef * vl1 - args.beta * ent
            optimizer.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            optimizer.step()

        return {
            'loss': loss.item(),
            'policy_loss': pl1.item(),
            'value_loss': vl1.item(),
            'entropy_mean': ent.item(),
            'grad_norm': gn.item(),
            'returns_mean': ret.mean().item(),
            'returns_std': ret.std().item(),
            'action_0_freq': (a_t == 0).float().mean().item(),
            'action_1_freq': (a_t == 1).float().mean().item(),
            'action_2_freq': (a_t == 2).float().mean().item(),
        }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Snake RL single-script training")

    # Game
    parser.add_argument("--grid-width", type=int, default=11)
    parser.add_argument("--grid-height", type=int, default=11)
    parser.add_argument("--vision-radius", type=int, default=5)
    parser.add_argument("--max-hunger-steps", type=int, default=88)
    parser.add_argument("--apple-speed", type=float, default=0.1)
    parser.add_argument("--num-snakes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Training
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--hidden1", type=int, default=128)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--talk-size", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=10000)

    # MLflow
    parser.add_argument("--experiment-name", type=str, default="train")
    parser.add_argument("--mlflow-uri", type=str, default="file:///tmp/mlruns")

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Vision size: number of cells in diamond of radius r = 2r(r+1)+1, +1 for head skipped
    # State matrix rows = visible_cells_excluding_head + 2 = 2r(r+1) + 2 = vision_size
    vision_radius = args.vision_radius
    vision_display_size = 2 * vision_radius + 1
    vision_size = 2 * vision_radius * (vision_radius + 1) + 2  # matches inf.py formula
    obs_size = vision_size * 2
    id_size = 2 if args.num_snakes == 2 else 0
    talk_size = args.talk_size if args.num_snakes == 2 else 0

    model = SnakeNet(
        input_size=obs_size + id_size,
        talk_size=talk_size,
        hidden_units_1=args.hidden1,
        hidden_units_2=args.hidden2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    env = SnakeGame(
        GRID_WIDTH=args.grid_width,
        GRID_HEIGHT=args.grid_height,
        VISION_RADIUS=vision_radius,
        VISION_DISPLAY_COLS=vision_display_size,
        VISION_DISPLAY_ROWS=vision_display_size,
        max_lifetime=10000,
        max_hunger_steps=args.max_hunger_steps,
        apple_speed=args.apple_speed,
        num_snakes=args.num_snakes,
    )

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    checkpoint_dir = "/tmp/snake_rl_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params(vars(args))

        for episode in range(1, args.max_episodes + 1):
            replay_buffer, stats = collect_episode(env, model, args)
            metrics = train_on_episode(replay_buffer, model, optimizer, args)

            log = {'apples': stats['apples'], 'steps': stats['steps']}
            if metrics:
                log.update(metrics)

            mlflow.log_metrics(log, step=episode)

            if episode % 100 == 0:
                print(f"[{episode}/{args.max_episodes}] apples={stats['apples']} "
                      f"steps={stats['steps']} "
                      f"loss={metrics['loss']:.4f}" if metrics else "")

            if episode % 50 == 0:
                ckpt = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pth")
                torch.save({
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args),
                }, ckpt)


if __name__ == "__main__":
    main()
