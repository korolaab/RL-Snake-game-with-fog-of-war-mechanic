import matplotlib.pyplot as plt
import numpy as np
import os

base = "/home/korolaab/projects/snake_rl/services/logs-storage/mlruns/511484317088474177"
runs = {
    "1 snake (14-12)": (os.path.join(base, "eec3a80407b1453d84cbe083340ad4c5/metrics"), "green", "lightgreen"),
    "2 snakes (14-12)": (os.path.join(base, "01035e22efc44b91bc9263d9c9ec70c9/metrics"), "red", "lightsalmon"),
    "2 snakes (64-32)": (os.path.join(base, "98d1b067e3ab4c6db137c35b1e950e6f/metrics"), "blue", "lightblue"),
}

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
fig.suptitle("Effect of network size — 2 snakes, hunger=88, 3000 episodes", fontsize=14, fontweight='bold')

metrics = [
    ("eaten_apples", "Eaten Apples per Episode", "Eaten Apples"),
    ("snake_length", "Max Snake Length per Episode", "Snake Length"),
    ("episode_length", "Episode Length (frames)", "Frames"),
    ("episode_reward", "Episode Reward", "Sum Reward"),
]

for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
    for label, (run_dir, color, light_color) in runs.items():
        try:
            steps, vals = read_metric(run_dir, metric)
            ax.bar(steps, vals, width=1.0, color=light_color, alpha=0.2)
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
out = "/home/korolaab/projects/snake_rl/services/bigger_net_comparison.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
