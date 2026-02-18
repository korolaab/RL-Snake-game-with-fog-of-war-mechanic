import matplotlib.pyplot as plt
import numpy as np
import os

run_dir = "/home/korolaab/projects/snake_rl/services/logs-storage/mlruns/511484317088474177/eec3a80407b1453d84cbe083340ad4c5/metrics"

def read_metric(name):
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
fig.suptitle("1 Snake — max_hunger_steps=88, 3000 episodes", fontsize=14, fontweight='bold')

configs = [
    ("eaten_apples", "Eaten Apples per Episode", "Eaten Apples", "red", "lightsalmon"),
    ("snake_length", "Max Snake Length per Episode", "Snake Length", "green", "lightgreen"),
    ("episode_length", "Episode Length (frames)", "Frames", "blue", "lightblue"),
    ("episode_reward", "Episode Reward", "Sum Reward", "orange", "moccasin"),
]

for ax, (metric, title, ylabel, color, light_color) in zip(axes.flat, configs):
    try:
        steps, vals = read_metric(metric)
        ax.bar(steps, vals, width=1.0, color=light_color, alpha=0.5)
        if len(vals) > 50:
            ma = moving_avg(vals, 50)
            ax.plot(steps[49:], ma, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    except FileNotFoundError:
        ax.set_title(f"{title} (not found)")

plt.tight_layout()
out = "/home/korolaab/projects/snake_rl/services/single_snake_hunger88.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
