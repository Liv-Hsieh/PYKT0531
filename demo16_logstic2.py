import matplotlib.pyplot as plt
import numpy as np

w1 = 0.5
w2 = 1.0
w3 = 2.0
w4 = 4.0
x = np.arange(-10, 10, 0.1)

for w in [w1, w2, w3, w4]:
    y = 1 / (1 + np.exp(-x * w))
    plt.plot(x,y,label=f'y=1/1+exp(-{w}x)')
plt.legend()
plt.show()
