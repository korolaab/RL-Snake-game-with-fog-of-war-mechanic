#!/usr/bin/env python3
"""Plot comparison of apple speed experiments from MLflow logs.

Usage:
    # First, copy logs from each pod:
    for speed in 0 0.1 0.2 0.3 0.5 0.7 1.0; do
        release="apple-speed-$(echo $speed | tr '.' '-')"
        pod=$(kubectl get pods -l job-name=${release}-snake-rl -o jsonpath='{.items[0].metadata.name}')
        mkdir -p logs_apple_speed/${speed}
        kubectl cp ${pod}:/logs/mlruns logs_apple_speed/${speed}/mlruns
    done

    # Then run this script:
    python scripts/plot_apple_speed_results.py --logs-dir logs_apple_speed
"""

import argparse
import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np


def read_mlflow_metric(run_dir, metric_name):
    """Read a metric's history from MLflow file store."""
    metric_file = os.path.join(run_dir, "metrics", metric_name)
    if not os.path.exists(metric_file):
        return [], []
    steps, values = [], []
    with open(metric_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                # Format: timestamp value step
                values.append(float(parts[1]))
                steps.append(int(parts[2]))
    return steps, values


def moving_average(values, window=100):
    """Compute moving average with given window."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def find_run_dir(experiment_dir):
    """Find the first run directory inside an MLflow experiment."""
    for exp_id in os.listdir(experiment_dir):
        exp_path = os.path.join(experiment_dir, exp_id)
        if not os.path.isdir(exp_path) or exp_id == ".trash":
            continue
        for run_id in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_id)
            if os.path.isdir(run_path) and os.path.exists(
                os.path.join(run_path, "metrics")
            ):
                return run_path
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot apple speed study results")
    parser.add_argument(
        "--logs-dir",
        default="logs_apple_speed",
        help="Directory containing per-speed MLflow logs",
    )
    parser.add_argument(
        "--metric", default="snake_length", help="Metric to plot"
    )
    parser.add_argument("--window", type=int, default=100, help="Moving average window")
    parser.add_argument("--output", default="apple_speed_comparison.png", help="Output file")
    args = parser.parse_args()

    speeds = sorted(
        [
            d
            for d in os.listdir(args.logs_dir)
            if os.path.isdir(os.path.join(args.logs_dir, d))
        ],
        key=float,
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(speeds)))

    for speed, color in zip(speeds, colors):
        mlruns_dir = os.path.join(args.logs_dir, speed, "mlruns")
        if not os.path.exists(mlruns_dir):
            print(f"Warning: no mlruns for speed={speed}, skipping")
            continue

        run_dir = find_run_dir(mlruns_dir)
        if run_dir is None:
            print(f"Warning: no run found for speed={speed}, skipping")
            continue

        steps, values = read_mlflow_metric(run_dir, args.metric)
        if not values:
            print(f"Warning: no {args.metric} data for speed={speed}, skipping")
            continue

        ma = moving_average(np.array(values), args.window)
        ma_steps = steps[args.window - 1 :] if len(steps) >= args.window else steps
        ax.plot(ma_steps, ma, label=f"speed={speed}", color=color, linewidth=1.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"{args.metric} (MA-{args.window})")
    ax.set_title(f"Apple Speed Study: {args.metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
