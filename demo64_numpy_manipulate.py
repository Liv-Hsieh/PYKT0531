import numpy as np

x1 = np.array([[1, 2], [3, 4]])
x2 = np.array([[5, 6], [7, 8]])
print(np.concatenate((x1, x2), axis=0))
print(np.concatenate((x1, x2), axis=1))
print(np.r_[x1, x2])
print(np.c_[x1, x2])
print('hstack')
print(np.hstack(x1))
print(np.hstack(x2))
print('vertical')
print(np.vstack(x1))
print(np.vstack(x2))