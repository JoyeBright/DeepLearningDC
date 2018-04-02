# Scaling up tpo multiple data points
import numpy as np
import Dataset4
from sklearn.metrics import mean_squared_error


def relu(x):
    out = max(0, x)
    return out


def predict_with_network(inputd, weights):

    node0_input = (inputd * weights['node0']).sum()
    node0_output = relu(node0_input)

    node1_input = (inputd * weights['node1']).sum()
    node1_output = relu(node1_input)

    hidden_layer_values = np.array([node0_output, node1_output])

    output = (hidden_layer_values * weights['output']).sum()
    return output


# Create model_output_0
model_output_0 = []

# Create model_output_1
model_output_1 = []

for row in Dataset4.input_data:
    model_output_0.append(predict_with_network(row, Dataset4.weights))
    model_output_1.append((predict_with_network(row, Dataset4.NewWeights)))

# Calculate the Mean Squared Error for model_output_0: mse0
mse0 = mean_squared_error(model_output_0, Dataset4.ActualTargets)

# Calculate the Mean Squared Error for model_output_1: mse1
mse1 = mean_squared_error(model_output_1, Dataset4.ActualTargets)

print("Mean Squared Error with first series of weights: %f" % mse0)
print("Mean Squared Error with new weights: %f" % mse1)
