import matplotlib.pyplot as plt
import numpy as np
import os

base = "/home/korolaab/projects/snake_rl/services/logs-storage/mlruns/511484317088474177"
runs = {
    "talk (dropout=0.0)": (os.path.join(base, "77a41a10838144cd9ec47a3d0e08f661/metrics"), "blue", "lightblue"),
    "no talk (dropout=1.0)": (os.path.join(base, "7baadf3e8d994e988aea3a5432097b47/metrics"), "red", "lightsalmon"),
}

def read_metric(run_dir, name):
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
fig.suptitle("Talk vs No Talk — 2 snakes, net 128-64, hunger=88, 10k episodes", fontsize=14, fontweight='bold')

metrics = [
    ("eaten_apples", "Eaten Apples per Episode", "Eaten Apples"),
    ("snake_length", "Max Snake Length per Episode", "Snake Length"),
    ("episode_length", "Episode Length (frames)", "Frames"),
    ("entropy_mean", "Entropy Mean", "Entropy"),
]

for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
    for label, (run_dir, color, light_color) in runs.items():
        try:
            steps, vals = read_metric(run_dir, metric)
            if len(vals) > 100:
                ma = moving_avg(vals, 100)
                ax.plot(steps[99:], ma, color=color, linewidth=2, label=label)
        except FileNotFoundError:
            pass
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "/home/korolaab/projects/snake_rl/services/10k_talk_vs_notalk.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
