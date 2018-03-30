import numpy as np

# The following input data are like number of children, num of accounts and so on
# And weights are also related to hidden layer and output layer
input_data = np.array([5, 7, 8, 1])

weights = {'node0': np.array([1, 1]),
           'node1': np.array([-1, 1]),
           'output': np.array([2, -1])}
