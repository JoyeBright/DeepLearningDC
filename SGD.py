import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import SGD
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


def get_new_model(input_s=n_cols):

    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(input_s,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model


lr_to_test = [.000001, 0.01, 1]

for lr in lr_to_test:
    print('\n\nTesting model with Learning Rate: %f\n' % lr)
    # Build new model to test, unaffected by previous models
    model_new = get_new_model()
    # Create SGD optimizer with specified learning rate
    my_optimizer = SGD(lr=lr)
    # Compile the model
    model_new.compile(optimizer=my_optimizer, loss='categorical_crossentropy')
    # Fit the model
    model_new.fit(predictors, target)
