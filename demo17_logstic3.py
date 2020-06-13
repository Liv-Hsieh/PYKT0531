import matplotlib.pyplot as plt
import numpy as np

w = 3
b1 = -20
b2 = 0
b3 = 8

x = np.arange(-10, 10, 0.1)

for b in [b1, b2, b3]:
    y = 1 / (1 + np.exp(-(x * w + b)))
    plt.plot(x, y, label=f'y=1/1+exp(-3x+{b})')
plt.legend()
plt.show()
