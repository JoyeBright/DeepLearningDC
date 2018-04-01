import numpy as np
import Dataset3 as Ds


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


ActualTarget = 3
model_output_0 = predict_with_network(Ds.input_data, Ds.weights)
error_0 = (model_output_0 - ActualTarget)

model_output_1 = predict_with_network(Ds.input_data, Ds.NewWeights)
error_1 = (model_output_1 - ActualTarget)

print("Model0 output:%d and Model1 output:%d" % (model_output_0, model_output_1))
print("Model0 error:%d and Model1 error:%d" % (error_0, error_1))
