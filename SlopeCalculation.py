# Slope Calculation Using Gradient Descent
import numpy as np

input_data = np.array([1, 3])
weights = np.array([1, 2])
ActualValue = 7

preds = (input_data * weights).sum()

error = preds - ActualValue

slope = 2 * error * input_data

print(slope)
