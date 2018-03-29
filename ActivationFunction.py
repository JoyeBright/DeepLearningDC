# Forward Propagation Algorithm By Using Activation Function
import numpy as np

# Define rectified linear unit function


def relu(inp):
    out = max(0, inp)
    return out


input_data = np.array([2,3])

# Using dictionary in order to save weights of hidden and output layer
weights = { 'node0': np.array([1,1]),
            'node1': np.array([-1,1]),
            'output': np.array([2,-1])}
node0_input = (input_data * weights['node0']).sum()
node0_output = relu(node0_input)
# Note: sum() is a built-in function which works as an iterator
node1_input = (input_data * weights['node1']).sum()
node1_output = relu(node1_input)

hidden_layer_values = np.array([node0_output,node1_output])
print("Hidden layers values: %s" % hidden_layer_values)
output = (hidden_layer_values * weights['output']).sum()

# Written transaction because of the problem
# Here we wanted to predict the number of the next year transaction based on two parameters or features
# Like age or number of children and so on
print("Total # of Transactions:%d" % output)

# Note: without activation function, you would have predicted a negative number!

