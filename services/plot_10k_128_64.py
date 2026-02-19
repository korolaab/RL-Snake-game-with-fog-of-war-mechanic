import matplotlib.pyplot as plt
import numpy as np
import os

base = "/home/korolaab/projects/snake_rl/services/logs-storage/mlruns/511484317088474177"
run_dir = os.path.join(base, "77a41a10838144cd9ec47a3d0e08f661/metrics")

def read_metric(name):
    steps, vals = [], []
    with open(os.path.join(run_dir, name)) as f:
        for line in f:
            parts = line.strip().split()
            vals.append(float(parts[1]))
            steps.append(int(parts[2]))
    return np.array(steps), np.array(vals)

def moving_avg(vals, window=100):
    return np.convolve(vals, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("2 Snakes, net 128-64, talk=12, hunger=88 — 10k episodes", fontsize=14, fontweight='bold')

configs = [
    ("eaten_apples", "Eaten Apples per Episode", "Eaten Apples", "red", "lightsalmon"),
    ("snake_length", "Max Snake Length per Episode", "Snake Length", "green", "lightgreen"),
    ("episode_length", "Episode Length (frames)", "Frames", "blue", "lightblue"),
    ("entropy_mean", "Entropy Mean", "Entropy", "purple", "plum"),
]

for ax, (metric, title, ylabel, color, light_color) in zip(axes.flat, configs):
    try:
        steps, vals = read_metric(metric)
        ax.bar(steps, vals, width=1.0, color=light_color, alpha=0.3)
        if len(vals) > 100:
            ma = moving_avg(vals, 100)
            ax.plot(steps[99:], ma, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    except FileNotFoundError:
        ax.set_title(f"{title} (not found)")

plt.tight_layout()
out = "/home/korolaab/projects/snake_rl/services/10k_128_64_results.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
