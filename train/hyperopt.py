#!/usr/bin/env python3
"""Hyperparameter optimization for Snake RL using Optuna."""

import argparse
import os
import sys

import numpy as np
import optuna
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services', 'Env', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from game import SnakeGame
from train import SnakeNet, collect_episode, train_on_episode


GRID_WIDTH = 11
GRID_HEIGHT = 11
VISION_RADIUS = 5
MAX_HUNGER = 88
APPLE_SPEED = 0.0
NUM_SNAKES = 1
EVAL_EPISODES = 1000   # episodes per trial
EVAL_WINDOW = 200      # last N episodes for objective


def make_args(trial, seed=42):
    vision_cells = 2 * VISION_RADIUS * (VISION_RADIUS + 1)
    obs_size = (vision_cells + 3) * 4

    lr          = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    beta        = trial.suggest_float("beta", 0.01, 0.5, log=True)
    gamma       = trial.suggest_float("gamma", 0.8, 0.99)
    gae_lambda  = trial.suggest_float("gae_lambda", 0.8, 0.99)
    hidden1     = trial.suggest_categorical("hidden1", [64, 128, 256])
    hidden2     = trial.suggest_categorical("hidden2", [32, 64, 128])
    value_coef  = trial.suggest_float("value_coef", 0.1, 1.0)
    ppo_epochs  = trial.suggest_int("ppo_epochs", 2, 8)

    class Args:
        pass

    a = Args()
    a.vision_radius = VISION_RADIUS
    a.num_snakes = NUM_SNAKES
    a.talk_size = 0
    a.max_hunger_steps = MAX_HUNGER
    a.apple_speed = APPLE_SPEED
    a.grid_width = GRID_WIDTH
    a.grid_height = GRID_HEIGHT
    a.apple_ttl = 0
    a.learning_rate = lr
    a.beta = beta
    a.gamma = gamma
    a.gae_lambda = gae_lambda
    a.hidden1 = hidden1
    a.hidden2 = hidden2
    a.value_coef = value_coef
    a.ppo_epochs = ppo_epochs
    a.eps_clip = 0.2
    a.obs_size = obs_size
    a.seed = seed
    return a


def objective(trial):
    args = make_args(trial)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vision_display_size = 2 * VISION_RADIUS + 1
    model = SnakeNet(
        input_size=args.obs_size,
        hidden_units_1=args.hidden1,
        hidden_units_2=args.hidden2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    env = SnakeGame(
        GRID_WIDTH=GRID_WIDTH, GRID_HEIGHT=GRID_HEIGHT,
        VISION_RADIUS=VISION_RADIUS,
        VISION_DISPLAY_COLS=vision_display_size,
        VISION_DISPLAY_ROWS=vision_display_size,
        max_hunger_steps=MAX_HUNGER,
        apple_speed=APPLE_SPEED,
        num_snakes=NUM_SNAKES,
    )

    apple_history = []

    for episode in range(1, EVAL_EPISODES + 1):
        replay_buffer, stats = collect_episode(env, model, args)
        train_on_episode(replay_buffer, model, optimizer, args)
        apple_history.append(stats['apples'])

        # Pruning: report intermediate value every 100 episodes
        if episode % 100 == 0:
            intermediate = float(np.mean(apple_history[-100:]))
            trial.report(intermediate, episode)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return float(np.mean(apple_history[-EVAL_WINDOW:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--storage", type=str,
                        default=None)
    parser.add_argument("--study-name", type=str, default="snake_single")
    args = parser.parse_args()

    if args.storage and args.storage.startswith("sqlite:///"):
        db_path = args.storage.replace("sqlite:///", "").lstrip("/")
        os.makedirs(os.path.dirname(f"/{db_path}") or ".", exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=300),
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    print("\n=== Best trial ===")
    t = study.best_trial
    print(f"  Objective: {t.value:.3f} apples")
    for k, v in t.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
