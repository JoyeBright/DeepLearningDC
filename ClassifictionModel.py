import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

source = pd.read_csv("Datasets/titanic_all_numeric.csv", delimiter=",")
# Source file times one convert True/False into 1/0
source = source * 1
# Convert DataFrame into the numpy matrix
source = source.as_matrix()
# Select whole of the columns except for the target value
predictors = np.delete(source, 0, 1)
# Number of the features or attributes
n_cols = predictors.shape[1]
# Select the first column which contained target values
target = source[:, 0]
# You can also obtain survived (target variable) by using source.survived
# Convert the target to categorical target
target = to_categorical(target)
# Set up the model
model = Sequential()
# Add the first layer
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
# Add the output layer
model.add(Dense(2, activation='softmax'))
# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# Fit the model
model.fit(predictors, target, epochs=10)
