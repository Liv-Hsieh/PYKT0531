import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a.view()
c = a

print(a, b, c, sep='\n')
print("now change b to (4,-1)")
b.shape = (4, -1)  # (4,1)
print(a, b, c, sep='\n')
print("now change c to (-1,4)")
c.shape = (-1, 4)
print(a, b, c, sep='\n')
