import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
import joblib
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt



model=load_model("models/mnist_pca_model_8.h5")

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()


X_train=X_train.reshape(-1, 28*28).astype("float32") / 255.0
X_test=X_test.reshape(-1, 28*28).astype("float32") / 255.0

pca=PCA(n_components=16)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)

#summary
model.summary()

weights_1= model.layers[0].get_weights()
weights_2= model.layers[1].get_weights()

w=weights_1[0]
b=weights_1[1]


W1, b1 = model.layers[0].get_weights()

print("Weights shape:", W1.shape)
print("Bias shape:", b1.shape)


x_manual = np.array([
    0.12, -0.35, 1.02, 0.48,
    -0.77, 0.11, 0.56, -0.22,
    0.93, -0.14, 0.07, 0.31,
    -0.41, 0.25, -0.66, 0.89
], dtype=np.float32)


sample_index = 44  # CHNAGE THIS ONE TO TEST DIFFEREN DIGIT SAMPLES
x_manual=X_test_pca[sample_index].astype(np.float32)

print("Input to first layer (x_manual):", x_manual.shape)
print("Input values:", (x_manual*100).astype(int))


z1 = np.dot(x_manual, W1) + b1
h1 = np.maximum(0, z1) #Relu activation


print("z1 shape:", z1.shape) 
print("h1 shape:", h1.shape)

print("Pre-activation z1:")
print(z1)

print("Hidden output after ReLU h1:")
print(h1)

last_layer_model= keras.Sequential([
    layers.Input(shape=(8,)), #change the input shape to match h1
    layers.Dense(10, activation="softmax")
])

W2, b2 = model.layers[1].get_weights()
last_layer_model.layers[0].set_weights([W2, b2])

h1_batch = np.expand_dims(h1, axis=0)   # shape (1, 16)
print("Input to last layer (h1):", h1_batch.shape)
y_pred_split = last_layer_model.predict(h1_batch, verbose=0)

print("Output probabilities from split model:")
print(y_pred_split[0])

print("Predicted class:")
print(np.argmax(y_pred_split[0]))


#quantization for analog mapping
scale=100

W1_int=np.round(W1*scale).astype(int)
b1_int=np.round(b1*scale).astype(int)


print("\nInteger W1:")
print(W1_int)

print("\nInteger b1:")
print(b1_int)

plt.imshow(X_test[sample_index].reshape(28, 28), cmap="gray")
plt.title(f"True label: {y_test[sample_index]}")
plt.axis("off")
plt.show()