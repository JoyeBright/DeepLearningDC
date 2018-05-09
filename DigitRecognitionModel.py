from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
history = model.fit(X, y, batch_size=128, epochs=20, verbose=1, validation_split=0.4)

