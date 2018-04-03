# Slope Calculation Using Gradient descent and multiplying to learning rate
import numpy as np

input_data = np.array([1, 2, 3])
weights = np.array([0, 2, 1])
learning_rate = 0.01
target = 0

preds = (weights * input_data).sum()

error = preds - target

slope = 2 * error * input_data

# I have read somewhere that we have not always to check the slope by subtracting the current weights. It is sometimes
# needed to add the value to the current weights. It is like making the gravity called gradient descent
# Update the weights: weights_updated
UpdatedWeights = weights - (slope * learning_rate)

UpdatedPreds = (UpdatedWeights * input_data).sum()

UpdatedError = UpdatedPreds - target

print("The error without applying gradient descent: %f" % error)
print("the error with applying gradient descent/UpdatedError: %f" % UpdatedError)


