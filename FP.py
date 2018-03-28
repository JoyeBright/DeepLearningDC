# Forward Propagation Algorithm
import numpy as np

input_data = np.array([2,3])

# Using dictionary in order to save weights of hidden and output layer
weights = { 'node0': np.array([1,1]),
            'node1': np.array([-1,1]),
            'output': np.array([2,-1])}
node0_value = (input_data * weights['node0']).sum()
# Note: sum() is a built-in function which works as an iterator
node1_value = (input_data * weights['node1']).sum()

hidden_layer_values = np.array([node0_value,node1_value])
print("Hidden layers values: %s" % hidden_layer_values)
output = (hidden_layer_values * weights['output']).sum()

# Written transaction because of the problem
# Here we wanted to predict the number of the next year transaction based on two parameters or features
# Like age or number of children and so on
print("Total # of Transactions:%d" % output)



