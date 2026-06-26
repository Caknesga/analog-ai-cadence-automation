import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
import joblib
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


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



