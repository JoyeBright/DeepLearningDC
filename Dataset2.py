import numpy as np

# The following input data are like number of children, num of accounts and so on
# And weights are also related to hidden layer and output layer
input_data = np.array([5, 7])

weights = {'node00': np.array([2, 4]),
           'node01': np.array([4, -5]),
           'node10': np.array([-1, 1]),
           'node11': np.array([2, 2]),
           'output': np.array([2, -1])}


