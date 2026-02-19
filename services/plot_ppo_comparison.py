import re
import numpy as np
import matplotlib.pyplot as plt

def parse_log(path):
    episodes, snake_lengths, apples, frames_list, rewards, entropy = [], [], [], [], [], []
    pattern = re.compile(
        r'\[Clock\] (\d+):snake_length=(\d+) eaten_apples=(\d+) .*entropy_mean=([\d.]+).* frames=(\d+)\s+sum_reward=(\d+)'
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                episodes.append(int(m.group(1)))
                snake_lengths.append(int(m.group(2)))
                apples.append(int(m.group(3)))
                entropy.append(float(m.group(4)))
                frames_list.append(int(m.group(5)))
                rewards.append(int(m.group(6)))
    return episodes, snake_lengths, apples, frames_list, rewards, entropy

def moving_avg(data, window=200):
    return np.convolve(data, np.ones(window)/window, mode='valid')

no_talk = parse_log("/tmp/claude-1000/-home-korolaab-projects-snake-rl/tasks/bbbae17.output")
talk = parse_log("/tmp/claude-1000/-home-korolaab-projects-snake-rl/tasks/b561adc.output")

window = 200

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('PPO Dual Snake 11x11: No Talk (dropout=1.0) vs Talk (dropout=0.0) — 10k episodes', fontsize=14)

metrics = [
    ('Eaten Apples', 2),
    ('Max Snake Length', 1),
    ('Episode Length (frames)', 3),
    ('Entropy', 5),
]

for ax, (title, di) in zip(axes.flat, metrics):
    n_data = no_talk[di]
    t_data = talk[di]
    n_ep = no_talk[0]
    t_ep = talk[0]

    ax.plot(n_ep, n_data, alpha=0.06, color='tab:blue')
    ax.plot(t_ep, t_data, alpha=0.06, color='tab:red')

    n_ma = moving_avg(n_data, window)
    t_ma = moving_avg(t_data, window)

    ax.plot(n_ep[window-1:], n_ma, color='tab:blue', linewidth=2, label='no talk (dropout=1.0)')
    ax.plot(t_ep[window-1:], t_ma, color='tab:red', linewidth=2, label='talk (dropout=0.0)')

    ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.legend(fontsize=8)

plt.tight_layout()
out = '/home/korolaab/projects/snake_rl/services/ppo_talk_comparison.png'
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
