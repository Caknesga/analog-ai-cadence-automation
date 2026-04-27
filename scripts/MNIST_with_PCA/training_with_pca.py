import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
import joblib
from sklearn.neural_network import MLPClassifier




(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

print(y_train)

X_train=X_train.reshape(-1, 28*28).astype("float32") / 255.0
X_test=X_test.reshape(-1, 28*28).astype("float32") / 255.0

pca=PCA(n_components=16)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)


print("Original shape:", X_train.shape)
print("PCA shape:", X_train_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", np.sum(pca.explained_variance_ratio_))


model= keras.Sequential([
    layers.Input(shape=(16,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(10, activation="softmax")
])


#model compiling
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history=model.fit(X_train_pca,y_train, epochs=10, batch_size=32, validation_split=0.1)

#evaulate
test_loss, test_acc=model.evaluate(X_test_pca,y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")


#summary
model.summary()

weights= model.layers[1].get_weights()

w=weights[0]
b=weights[1]

print("Weights shape:", w.shape)
for weight in w:
    print(weight)

#for layer in model.layers:
#    print(layer.name)
#    print(layer.get_weights())


model.save("models/mnist_pca_model_8.h5")

