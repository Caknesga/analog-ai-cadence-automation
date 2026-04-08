import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

print(y_train)


plt.imshow(X_train[1])
plt.show(   )
