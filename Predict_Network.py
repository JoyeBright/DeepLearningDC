import numpy as np
# The following Dataset imported from external python file
import Dataset1


def relu(x):
    out = max(0, x)
    return out


def predict_with_network(in_data_row, weights):
    # np.set_printoptions()
    print("\nFirst element of input data: %d" % in_data_row)
    print("Weights for node0: %s" % weights['node0'])
    print("Weights for node1: %s" % weights['node1'])
    print("Weights for output: %s" % weights['output'])
    # Calculate node 0 value
    node_0_input = (in_data_row * weights['node0']).sum()
    print("Node 0 in hidden layer before activation function: %d" % node_0_input)
    node_0_output = relu(node_0_input)
    print("Node 0 in hidden layer after activation function: %d" % node_0_output)

    # Calculate node 1 value
    node_1_input = (in_data_row * weights['node1']).sum()
    print("Node 1 in hidden layer before activation function: %d" % node_1_input)
    node_1_output = relu(node_1_input)
    print("Node 1 in hidden layer after activation function: %d" % node_1_output)

    # Put node values into array: hidden_layer_output
    hidden_layer_output = np.array([node_0_output,node_1_output])
    print("Hidden layer: %s" % hidden_layer_output)

    # Calculate model output
    input_to_final_layer = (hidden_layer_output * weights['output']).sum()
    print("Output layer before activation function: %d" % input_to_final_layer)
    model_output = relu(input_to_final_layer)
    print("Output layer after activation function: %d" % model_output)

    # Return model output
    return model_output


# Create Empty list to store prediction results
results = []
for input_data_row in Dataset1.input_data:
    # Append prediction to result
    results.append(predict_with_network(input_data_row, Dataset1.weights))

# Print results
print(results)
