import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
import joblib
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import load_model
import subprocess
import os
import time


# Load the trained model
model=load_model("models/mnist_pca_model_8.h5")

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()


X_train=X_train.reshape(-1, 28*28).astype("float32") / 255.0
X_test=X_test.reshape(-1, 28*28).astype("float32") / 255.0

pca=PCA(n_components=16)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)

# extract weights and biases from the first layer

sample_index = 184  # CHNAGE THIS ONE TO TEST DIFFEREN DIGIT SAMPLES
x_manual=X_test_pca[sample_index].astype(np.float32)
inputs_values=(x_manual*100).astype(int)/200

print("Weights and biases of the first layer for the circuit mapping:   ")
print("\nInput to first layer (x_manual):", x_manual.shape)
print("\nInput values:", inputs_values)

W1, b1 = model.layers[0].get_weights()

print("\nWeights shape:", W1.shape)
print("\nBias shape:", b1.shape)


#quantization for analog mapping
scale=100

W1_int=np.round(W1*scale).astype(int)/20  #we devide 20 to fit the weights rnage in voltage headroom
b1_int=np.round(b1*scale).astype(int)/200  


first_neuron_Weight=W1_int[:,0]
first_neuron_Bias=b1_int[0]

#aboslute values for sign analysis
x_manual_abs=abs(inputs_values)   
W1_abs=abs(W1_int)
b1_abs=abs(b1_int)

first_neuron_Weight_abs=W1_abs[:,0]
first_neuron_Bias_abs=b1_abs[0]


print("\nFirst neuron weights:")
print(first_neuron_Weight)      
print("\nFirst neuron bias:")
print(first_neuron_Bias)

products=first_neuron_Weight * x_manual # element-wise, shape (16,)
S = np.where(products < 0, 1.5, 0.0) # 1.5V for negative products, 0V for non-negative products

print("=" * 50)
print("S_i VALUES — Neuron 0")
print("=" * 50)
print(f"\n{'i':<6} {'Input x_i':>12} {'Weight w_i':>12} {'w_i * x_i':>12} {'S_i':>6}")
print("-" * 52)
for i in range(16):
    prod = products[i]
    print(f"S_{i+1:<4} {x_manual_abs[i]:>12.6f} {first_neuron_Weight_abs[i]:>12.6f} {prod:>12.6f} {S[i]:>6.1f}")

print("-" * 52)
print(f"\nS vector: {S}")



z1 = np.dot(x_manual, W1) + b1
h1 = np.maximum(0, z1) #Relu activation


print("h1 shape:", h1.shape)

print("Hidden output after ReLU h1:")
print((h1*100).astype(int)/200)

last_layer_model= keras.Sequential([
    layers.Input(shape=(8,)), #change the input shape to match h1
    layers.Dense(10, activation="softmax")
])

W2, b2 = model.layers[1].get_weights()
last_layer_model.layers[0].set_weights([W2, b2])

h1_batch = np.expand_dims(h1, axis=0)   # shape (1, 16)
print("Input to last layer (h1):", h1_batch.shape)
y_pred_split = last_layer_model.predict(h1_batch, verbose=0)

#print("Output probabilities from split model:")
#print(y_pred_split[0])

print("Predicted class:")
print(np.argmax(y_pred_split[0]))

print("True class:")
print(y_test[sample_index])


#OCEAN SIMULATION PART 


def parse_eng(val):
    if isinstance(val, (int, float)):
        return float(val)
    val = val.strip()
    scale = {
        'f': 1e-15, 'p': 1e-12, 'n': 1e-9,
        'u': 1e-6,  'm': 1e-3,  'k': 1e3,
        'M': 1e6,   'G': 1e9,
    }
    suffix = val[-1]
    return float(val[:-1]) * scale[suffix] if suffix in scale else float(val)


# ============================================================
# YOUR WORKING OCEAN FILE — just update the desVar values
# ============================================================

def update_and_run_ocean(
    template_path   = "./cadence/ocean/transfer.ocn",   # your working ocean file
    output_csv      = "./SWAP_test1.csv",
    neuron_idx      = 0,
    x_input         = None,                 # shape (16,)
    weights         = None,                 # shape (16,) — W1_int[:, neuron_idx]
    signs           = None,                 # shape (16,) — S vector
    capacitors      = None,                 # shape (16,) — C values in Farads
):

    # Read your existing working ocean file as template
    with open(template_path, "r") as f:
        ocean_content = f.read()

    # --- Update desVar values in the ocean file ---
    import re

    def replace_desvar(content, name, value):
        """Replace a desVar value in the ocean file."""
        pattern = rf'(desVar\(\s*"{name}"\s*)([^\)]+)(\))'
        replacement = rf'\g<1>{value}\g<3>'
        return re.sub(pattern, replacement, content)

    # Update V values (input voltages)
    for i in range(16):
        ocean_content = replace_desvar(ocean_content, f"V{i+1}", f"{x_input[i]:.4f}")

    # Update C values (capacitors) — convert Farads to pF string
    for i in range(16):
        c_pf = capacitors[i] * 1e12
        ocean_content = replace_desvar(ocean_content, f"C{i+1}", f"{c_pf:.4f}p")

    # Update sign values
    for i in range(16):
        ocean_content = replace_desvar(ocean_content, f"s{i+1}", f"{signs[i]}")

    # Update output csv name per neuron so files don't overwrite each other
    neuron_csv = output_csv.replace(".csv", f"_neuron{neuron_idx}.csv")
    ocean_content = ocean_content.replace(output_csv, neuron_csv)

    # Write updated ocean file
    updated_path = f"./SWAP_neuron{neuron_idx}.ocn"
    with open(updated_path, "w") as f:
        f.write(ocean_content)

    print(f"[Neuron {neuron_idx}] Running Ocean simulation...")

    # --------------------------------------------------------
    # RUN YOUR OCEAN FILE WITH SUBPROCESS — same as before
    # --------------------------------------------------------
    result = subprocess.run(
        ["ocean", "-nograph", "-replay", updated_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"[ERROR] Neuron {neuron_idx} failed:\n{result.stderr}")
        return None

    # Read output from csv
    try:
        with open(neuron_csv, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        raw_val = lines[-1].split()[-1]
        output_voltage = parse_eng(raw_val)
        print(f"[Neuron {neuron_idx}] Output: {output_voltage:.6f} V")
        return output_voltage
    except Exception as e:
        print(f"[ERROR] Could not read output for neuron {neuron_idx}: {e}")
        return None

S_matrix = np.zeros((16, 8))

for neuron_idx in range(8):
    weights_n = W1_int[:, neuron_idx]          # (16,) weights for this neuron
    products  = weights_n * x_manual           # element-wise (16,)
    S_matrix[:, neuron_idx] = np.where(products < 0, 1.5, 0.0)

print("S_matrix shape:", S_matrix.shape)   # should be (16, 8)
print(S_matrix)

# ============================================================
# MAIN — loop over 8 neurons
# ============================================================

h1_circuit = np.zeros(8)

for neuron_idx in range(8):
    weights  = W1_int[:, neuron_idx]
    signs    = S_matrix[:, neuron_idx]        # now correctly (16,)
    caps     = np.abs(weights) * 1e-12

    output = update_and_run_ocean(
        template_path = "./cadence/ocean/transfer.ocn",
        output_csv    = "./SWAP_test2.csv",
        neuron_idx    = neuron_idx,
        x_input       = x_manual,
        weights       = weights,
        signs         = signs,
        capacitors    = caps,
    )

    if output is not None:
        h1_circuit[neuron_idx] = output

