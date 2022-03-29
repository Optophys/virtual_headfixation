# from wrapped_sfmtools.SfmWrapper import SfmTool
from wrapped_sfmtools.SfmWrapper import *

t = SfmTool()

# # use actual lists
# x_vec = [[0.5, 0.5], [1.0, 1.0]]
# P_vec = [[1.0, 0.0, 0.0,
#           0.0, 1.0, 0.0,
#           0.0, 0.0, 1.0,
#           0.0, 0.0, 0.0],
#
#          [1.0, 0.0, 0.0,
#           0.0, 1.0, 0.0,
#           0.0, 0.0, 1.0,
#           0.0, 0.0, -1.0]]
# result = t.triangulateLinear(P_vec, x_vec)
# print('looks okay')
# print(len(result))
# print(result)

# # use numpy arrays
# import numpy as np
#
# x_vec = [np.array([0.5, 0.5]), np.array([1.0, 1.0])]
# P_vec = [np.concatenate([np.eye(3), np.array([[0.0], [0.0], [0.0]])], 1),
#          np.concatenate([np.eye(3), np.array([[0.0], [0.0], [-1.0]])], 1)]
#
# x_vec = [x.flatten().tolist() for x in x_vec]
# P_vec = [x.flatten('F').tolist() for x in P_vec]
# print(x_vec)
# print()
# print(P_vec)
# result = t.triangulateLinear(P_vec, x_vec)
# print('looks okay')
# print(len(result))
# print(result)

# use numpy arrays
import numpy as np

x_vec = [np.array([0.5, 0.5]), np.array([1.0, 1.0])]
P_vec = [np.concatenate([np.eye(3), np.array([[0.0], [0.0], [0.0]])], 1),
         np.concatenate([np.eye(3), np.array([[0.0], [0.0], [-1.0]])], 1)]

result = t.triangulateLinear(P_vec, x_vec)
print('looks okay')
print(result)
