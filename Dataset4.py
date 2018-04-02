import numpy as np

input_data = np.array([0, 3])

weights = {'node0': np.array([2, 1]),
           'node1': np.array([1, 2]),
           'output': np.array([1, 1])}

NewWeights = {'node0': np.array([0, 1]),
              'node1': np.array([1, 0]),
              'output': np.array([1, 1])}


