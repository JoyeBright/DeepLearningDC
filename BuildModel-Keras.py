# In Pandas which is an open source BSD-licensed python library, easy to use data structures and data
# analysis tools for the python PL
# Pandas delase with three DS, Panel, Dataframe, series
# In Pandas DataFrame, .head(n=5) return the first n rows
# In Pandas DataFrame, .describe() generates descriptive statistics that summarize the central tendency,
# dispersion, shape of a dataset's distribution, exluding NaN (Not a number) values.

import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential

predictors = np.loadtxt('', delimiter=',')
n_cols = predictors.shape[1]
model = Sequential()
# Add the first layer
model.add(Dense(50, Activation="relu", input_shape=(n_cols,)))
# Add the second layer
model.add(Dense(32, Activation="relu"))
# Add the output layer
model.add(Dense(1))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# What is loss function of the method?
# By Printing model.loss u can access its loss function
# Fitting the model
model.fit(predictors, target)

