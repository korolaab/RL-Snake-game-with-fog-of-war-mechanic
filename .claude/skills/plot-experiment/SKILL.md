---
name: plot
description: Plot training results from MLflow experiments. Use when user asks to plot, show graph, visualize results, or check experiment progress.
argument-hint: "[experiment_name or mlflow_run_path]"
allowed-tools: Bash, Read
---

When invoked, do the following:

## Step 1 — Find the data

If $ARGUMENTS is provided, search for that experiment name in MLflow:
```bash
ssh COMP "find /data-storage/hdd2/snake_rl/experiments/mlruns -name 'meta.yaml' | xargs grep -l '$ARGUMENTS' 2>/dev/null | head -1"
```

If no argument, list recent experiments and pick the most recently updated one:
```bash
ssh COMP "find /data-storage/hdd2/snake_rl/experiments/mlruns -name 'apples' | xargs ls -t 2>/dev/null | head -5"
```

Get current episode count and mean of last 100:
```bash
ssh COMP "tail -1 <metrics/apples path> && tail -100 <metrics/apples path> | awk '{sum+=\$2} END {print \"mean last 100:\", sum/NR}'"
```

Download the full apples metric:
```bash
ssh COMP "cat <metrics/apples path>" | awk '{print $2}' > /tmp/plot_apples.txt
```

Also download `steps` metric if available for context.

## Step 2 — Build and show the plot

Run this Python snippet (use `DISPLAY=:0` and `matplotlib.use('TkAgg')` for display):

```python
import numpy as np, matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

apples = np.loadtxt('/tmp/plot_apples.txt')
window = min(200, len(apples) // 5)
smoothed = np.convolve(apples, np.ones(window)/window, mode='valid')
x_smooth = np.arange(window, len(apples) + 1)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(np.arange(1, len(apples)+1), apples, alpha=0.15, linewidth=0.5, label='per episode')
ax.plot(x_smooth, smoothed, linewidth=2, label=f'rolling mean {window}')
ax.axhline(smoothed[-1], color='red', linestyle='--', alpha=0.7, label=f'current avg: {smoothed[-1]:.1f}')
ax.set_xlabel('Episode'); ax.set_ylabel('Apples')
ax.set_title(f'<experiment_name> — {len(apples)} episodes')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()

# Save to project
import os, datetime
os.makedirs('experiments/plots', exist_ok=True)
fname = f"experiments/plots/<experiment_name>_{datetime.date.today()}.png"
plt.savefig(fname, dpi=150)
print(f"Saved: {fname}")
plt.show()

print(f"Max smoothed: {smoothed.max():.1f} at ep {x_smooth[smoothed.argmax()]}")
print(f"Final avg: {smoothed[-1]:.1f}")
```

## Step 3 — Display the saved image

Use the Read tool to show the saved PNG file inline.

## Step 4 — Write a summary

After seeing the plot, write a short analysis covering:
- **Текущий прогресс**: сколько эпизодов, среднее яблок
- **Тренд**: растёт / стабилизировалось / упало
- **Пик**: максимум когда был
- **Вывод**: на плато или есть потенциал, стоит ли продолжать обучение
