import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


def model_generator(number_of_unit):
    # Define a model
    model = Sequential()
    model.add(Dense(number_of_unit, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(number_of_unit, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model


model1 = model_generator(10)
model2 = model_generator(100)

# Compile the models
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Using EarlyStopping monitor callback
# patience=3 indicates how many epochs should pass without improving
# Imporoving here means less val_loss
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=3)
# Fit the model
model1_training = model1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)
model2_training = model2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)
# Create the plot
plt.plot(model1_training.history['val_loss'], 'r', model2_training.history['val_loss'], 'b')
plt.suptitle('Note: Model1 and Model2 have 2 hidden \nlayers with 10 and 100 units respectively', fontsize=11)
plt.xlabel('Epochs')
plt.ylabel('Validation Score')
plt.legend(('Model1', 'Model2'), loc='upper right')
plt.show()
