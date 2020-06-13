import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.1)
y = 1 / (1 + np.exp(-x))
plt.xlabel(x)
plt.ylabel(y)
plt.plot(x, y)
plt.show()
