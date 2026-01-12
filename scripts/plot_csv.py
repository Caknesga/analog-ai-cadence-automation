import numpy as np
import matplotlib.pyplot as plt

# Path to CSV
csv_path = "cadence/results/vouts.csv"

# Load data (space-separated, no header)
data = np.loadtxt(csv_path)

print("Data shape:", data.shape)

# Case 1: time + 4 outputs (most likely)
if data.shape[1] == 5:
    time  = data[:, 0]
    vouts = data[:, 1:]

# Case 2: only outputs (no time)
elif data.shape[1] == 4:
    time  = np.arange(len(data))
    vouts = data

else:
    raise ValueError("Unexpected CSV format")

# Plot
plt.figure()
for i in range(vouts.shape[1]):
    plt.plot(time, vouts[:, i], label=f"Vout{i+1}")

plt.xlabel("Time (s)" if data.shape[1] == 5 else "Sample index")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
