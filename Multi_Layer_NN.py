# In this exercise, you'll write code to do forward propagation for neural netwrok with 2
# hidden layers. Each hidden layer has two nodes.
# Nodes in first hidden layer are called node00 and node01 and their weights are loaded based on
# Dataset2 as weights['node00'] and weights['node01'] respectively.
# Nodes in second hidden layer are called node10 and node11 and their weights are loaded based on
# Dataset2 as weights['node10'] and weights['node11'] respectively.

from Dataset2 import weights, input_data
import numpy as np


def relu(x):
    out = max(0,x)
    return out


def predict_with_network(inputd):

    node00_input = (inputd * weights['node00']).sum()
    node00_output = relu(node00_input)

    node01_input = (inputd * weights['node01']).sum()
    node01_output = relu(node01_input)

    hidden_0_outputs = np.array([node00_output, node01_output])

    node10_input = (hidden_0_outputs * weights['node10']).sum()
    node10_output = relu(node10_input)

    node11_input = (hidden_0_outputs * weights['node11']).sum()
    node11_output = relu(node11_input)

    hidden_1_outputs = np.array([node10_output, node11_output])

    model_output = (hidden_1_outputs * weights['output']).sum()
    # This part of code has been added by me cause i think it would be better if output value came
    # negative consequently model output return zero instead of minus value
    model_output = relu(model_output)

    return model_output


output = predict_with_network(input_data)

print(output)




