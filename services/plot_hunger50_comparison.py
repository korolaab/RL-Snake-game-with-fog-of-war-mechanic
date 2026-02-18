import matplotlib.pyplot as plt
import numpy as np
import os

base = "/home/korolaab/projects/snake_rl/services/logs-storage/mlruns/511484317088474177"
talk_run = os.path.join(base, "feb8c762750241d8a491258b5c6ecca8/metrics")
notalk_run = os.path.join(base, "5dd91659d84d4614a42706ec179802cd/metrics")

def read_metric(run_dir, name):
    steps, vals = [], []
    with open(os.path.join(run_dir, name)) as f:
        for line in f:
            parts = line.strip().split()
            vals.append(float(parts[1]))
            steps.append(int(parts[2]))
    return np.array(steps), np.array(vals)

def moving_avg(vals, window=50):
    return np.convolve(vals, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Dual Snake: Talk vs No Talk — max_hunger_steps=50 (3000 episodes)", fontsize=14, fontweight='bold')

configs = [
    ("eaten_apples", "Eaten Apples per Episode", "Eaten Apples"),
    ("snake_length", "Max Snake Length per Episode", "Snake Length"),
    ("episode_length", "Episode Length (frames)", "Frames"),
    ("episode_reward", "Episode Reward", "Sum Reward"),
]

colors = {
    "talk": ("blue", "lightblue"),
    "notalk": ("red", "lightsalmon"),
}

for ax, (metric, title, ylabel) in zip(axes.flat, configs):
    for label, run_dir, (color, light_color) in [
        ("talk (dropout=0.0)", talk_run, colors["talk"]),
        ("no talk (dropout=1.0)", notalk_run, colors["notalk"]),
    ]:
        try:
            steps, vals = read_metric(run_dir, metric)
            ax.bar(steps, vals, width=1.0, color=light_color, alpha=0.3)
            if len(vals) > 50:
                ma = moving_avg(vals, 50)
                ax.plot(steps[49:], ma, color=color, linewidth=2, label=label)
        except FileNotFoundError:
            pass
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "/home/korolaab/projects/snake_rl/services/hunger50_comparison.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
