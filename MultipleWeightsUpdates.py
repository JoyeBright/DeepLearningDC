# Make multiple update
# This network does not have any hidden layers, and it goes directly from the input (3 nodes) to
# an output. Note that weights is a single array
import Dataset5
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

UpdateNum = 20
mse_hist = []


def get_slope(inputd, target, weights):
    preds = (inputd * weights).sum()
    error = preds - target
    out = 2 * error * inputd
    return out


def get_mse(inputd, target, updated_weights):
    preds = (inputd * updated_weights).sum()
    out = mean_squared_error(preds, target)
    return out


# Iterate over the number of updates
for i in range(UpdateNum):
    # Calculate the slope
    slope = get_slope(Dataset5.input_data, Dataset5.target, Dataset5.weights)
    # Update the weights
    UpdatedWeights = Dataset5.weights - (slope * Dataset5.learning_rate)
    # Calculate mse with new weights(Updated Weights)
    mse = get_mse(Dataset5.input_data, Dataset5.target, UpdatedWeights)
    # Append the mse to mse_hist[]
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.show()
