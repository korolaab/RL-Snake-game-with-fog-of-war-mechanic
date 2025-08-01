import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file; change 'data.csv' to your file name if needed.
data = pd.read_csv("score_history.csv")
# Calculate sliding average (rolling mean), adjust window size as needed
window_size = 50
data["rolling_avg"] = data["score"].rolling(window=window_size, min_periods=1).mean()

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Plot the original scores with transparency
sns.lineplot(data=data, x="episode", y="score", marker="o", alpha=0.2, label="Score")

# Plot the rolling average with a solid line
sns.lineplot(data=data, x="episode", y="rolling_avg", label=f"Rolling Avg (window={window_size})")

plt.title("Episode vs Score with Rolling Average")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig("plot.png")
print("Plot saved as plot.png")
