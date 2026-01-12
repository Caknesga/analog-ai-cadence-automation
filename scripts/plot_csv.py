import numpy as np
import matplotlib.pyplot as plt


def parse_eng(val):
 
    if isinstance(val, (int, float)):
        return float(val)

    val = val.strip()

    scale = {
        'f': 1e-15,
        'p': 1e-12,
        'n': 1e-9,
        'u': 1e-6,
        'm': 1e-3,
        'k': 1e3,
        'M': 1e6,
        'G': 1e9,
    }

    suffix = val[-1]
    if suffix in scale:
        return float(val[:-1]) * scale[suffix]
    else:
        return float(val)
    


rows = []

"""with open("_vouts.csv") as f:
    for line in f:
        if line.strip().startswith("#") or not line.strip():
            continue  # skip comments / empty lines
        parts = line.split()
        rows.append([parse_eng(p) for p in parts])"""


with open("_vouts.csv") as f:
    for i, line in enumerate(f):
        if i < 3:        # skip first 3 rows → data starts at row 4
            continue
        parts = line.split()
        rows.append([parse_eng(p) for p in parts])

data = np.array(rows)

print("Data shape:", data.shape)
print("First row:", data[0])

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
