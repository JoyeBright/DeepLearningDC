import numpy as np

input_data = np.array([0, 3, 5, 7, 8, 2])

weights = {'node0': np.array([2, 1]),
           'node1': np.array([1, 2]),
           'output': np.array([1, 1])}

NewWeights = {'node0': np.array([0, 1]),
              'node1': np.array([1, 0]),
              'output': np.array([1, 1])}

ActualTargets = [1, 2, 3, 3, 9, 3]
