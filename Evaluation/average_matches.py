import numpy as np
import matplotlib.pyplot as plt

# Read the file

title="matches_log_bruteforce.csv"
title_name = title.split(".")[0]
with open(title, "r") as f:
    raw_data = f.read()

    #omit the first line
    raw_data = raw_data.split("\n", 1)[1]


# Split executions by blank lines
executions = [list(map(int, block.split())) for block in raw_data.strip().split("\n\n")]

# Compute statistics for each execution
for i, matches in enumerate(executions, 1):
    mean_val = np.mean(matches)
    std_val = np.std(matches)
    min_val = np.min(matches)
    max_val = np.max(matches)
    print(f"Execution {i}: Mean={mean_val:.2f}, Std={std_val:.2f}, Min={min_val}, Max={max_val}")

# Compute average number of matches per execution
execution_means = [np.mean(matches) for matches in executions]
print(f"Average Matches: {np.mean(execution_means):.2f}")

# Plot the mean graph +- the standard deviation
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(execution_means) + 1), execution_means, marker='o', linestyle='-', color='b', label="Avg Matches")
plt.fill_between(range(1, len(execution_means) + 1), np.array(execution_means) - np.std(execution_means), np.array(execution_means) + np.std(execution_means), color='b', alpha=0.2, label="Std Dev")
plt.xlabel("Execution Number")
plt.ylabel("Average Matches")
plt.title("Average Number of Matches per Execution")

plt.xticks(range(1, len(execution_means) + 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(execution_means) + 1), execution_means, marker='o', linestyle='-', color='b', label="Avg Matches")
plt.xlabel("Execution Number")
plt.ylabel("Average Matches")
plt.title("Average Number of Matches per Execution")
plt.xticks(range(1, len(execution_means) + 1)) 
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.title(title_name)
plt.show()
