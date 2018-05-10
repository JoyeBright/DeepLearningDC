from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=np.nan, linewidth=200)

source = pd.read_csv("Datasets/mnist.csv", delimiter=",")
y = source.iloc[:, 0]
X = source.iloc[:, 1:].values
# Cast a pandas object to a specified dtype
X = X.astype('float32')
X = X/256
y = to_categorical(y, 10)

model = Sequential()

model.add(Dense(100, activation='relu', input_shape=(784, )))
# If I use input_dim=784, it is the same syntax
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit method equals to train in machine learning
history = model.fit(X, y, batch_size=128, epochs=10, verbose=1, validation_split=0.4)
# evaluate the network
# The net can be evaluated on a seperate dataset, unseen during testing.
# This will provide an estimate of the performance of the net prediction for unseen data in the future
loss, accuracy = model.evaluate(X, y)
print("\nLoss: %.2f, Accuracy: %.2f" % (loss, (accuracy*100)))
# Finally, once we are satisfied with the performance of our fit model, we can use it to make prediction on new data
# Model Prediction
newX = source.iloc[1993:, 1:].values
newX = newX.astype('float32')
newX = newX/256
probabilities = model.predict(newX)
# argmax return the indices of the maximum value along an axis
# Axis=1 due to finding max in each rows not each columns
print("Predicted Digits: %s" % (np.argmax(probabilities, axis=1)))

