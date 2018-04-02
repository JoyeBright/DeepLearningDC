# Make multiple update
# This network does not have any hidden layers, and it goes directly from the input (3 nodes) to
# an output. Note that weights is a single array
import numpy as np
import matplotlib.pyplot as plt


input_data = np.array([1, 2, 3])
target = 0
weights = np.array([0, 2, 1])
UpdateNum = 20
mse_hist = []


def get_error(inputd, target_actual, weight):
    preds = (inputd * weight).sum()
    error = preds - target_actual
    return error


def get_slope(inputd, target_actual, weight):
    error = get_error(inputd, target_actual, weight)
    out = 2 * error * inputd
    return out


def get_mse(inputd, target, updated_weights):
    errors = get_error(inputd, target, updated_weights)
    mseresult = np.mean(errors**2)
    return mseresult


# Iterate over the number of updates
for i in range(UpdateNum):
    # Calculate the slope
    slope = get_slope(input_data, target, weights)
    # Update the weights
    weights = weights - (slope * 0.01)
    # Calculate mse with new weights(Updated Weights)
    mse = get_mse(input_data, target, weights)
    # Append the mse to mse_hist[]
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.show()
