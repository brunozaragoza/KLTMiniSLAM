import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd


df = pd.read_csv('task_5_image_data_part2/results.csv')
data = df.to_dict(orient='list')

df = pd.DataFrame(data)

average_t_len = df['t_len'].mean()
average_rmse = df['rmse'].mean()

print("Trajectory Length (t_len) and Absolute Translation Error (ATE) for each run:")
for i in range(len(df)):
    print(f"Run {i+1}: t_len = {df['t_len'][i]}, ATE (rmse) = {df['rmse'][i]}")
    
print(f"\nAverage trajectory length: {average_t_len}")
print(f"Average ATE (rmse): {average_rmse}")



plt.figure(figsize=(8, 5))
plt.bar(range(len(df)), df['t_len'], color='b', label="Trajectory Length")
plt.axhline(y=average_t_len, color='r', linestyle='--', label="Average Trajectory Length")

plt.text(0, average_t_len, f'{average_t_len:.2f}', ha='right', va='bottom')
plt.xlabel("Run Number")
plt.ylabel("Trajectory Length")
plt.title("Trajectory Length per Run")
plt.xticks(range(len(df)))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(df) + 1), df['mean'], marker='o', linestyle='-', color='b', label="Mean ATE")
plt.fill_between(range(1, len(df) + 1), np.array(df['mean']) - np.array(df['std']), np.array(df['mean']) + np.array(df['std']), color='b', alpha=0.2, label="Std Dev")
plt.xlabel("Run Number")
plt.ylabel("Mean ATE")
plt.title("Mean ATE per Run")
plt.xticks(range(1, len(df) + 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(range(len(df)), df['rmse'], color='b', label="ATE (rmse)")
plt.axhline(y=average_rmse, color='r', linestyle='--', label="Average ATE (rmse)")
plt.text(0, average_rmse, f'{average_rmse:.5f}', ha='right', va='bottom')
plt.xlabel("Run Number")
plt.ylabel("ATE (rmse)")
plt.title("ATE (rmse) per Run")
plt.xticks(range(len(df)))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()


