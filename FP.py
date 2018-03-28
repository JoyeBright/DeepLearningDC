# Coding the forward propagation algorithm
import numpy as np

input_data = np.array([2,3])

#using dictionary in order to save weights of hidden and output layer
weights = { 'node0': np.array([1,1]),
            'node1': np.array([-1,1]),
            'output': np.array([2,-1])}
node0_value = (input_data * weights['node0']).sum()
#note: sum() is a built-in function which works as an iterator
node1_value = (input_data * weights['node1']).sum()

hidden_layer_values = np.array([node0_value,node1_value])
print("Hidden layers values: %s" %hidden_layer_values)
output = (hidden_layer_values * weights['output']).sum()

print("Total # of Transactions:%d" %output)



