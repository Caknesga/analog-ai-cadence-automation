import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense,Flatten


(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

print(y_train)

X_train=X_train.reshape(-1, 28*28).astype("float32") / 255.0
X_test=X_test.reshape(-1, 28*28).astype("float32") / 255.0

model= keras.Sequential([
    layers.Input(shape=(28*28,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])


#model compiling
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#train
history=model.fit(X_train,y_train, epochs=7, batch_size=32, validation_split=0.1)

#evaulate
test_loss, test_acc=model.evaluate(X_test,y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

#model.summary()

weights= model.layers[1].get_weights()

w=weights[0]
b=weights[1]

print("Weights shape:", w.shape)
for weight in w:
    print(weight)

#for layer in model.layers:
#    print(layer.name)
#    print(layer.get_weights())