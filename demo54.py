import numpy as np
import tensorflow as tf

a1 = np.arange(-10, 10)
print(type(a1), a1)
for a in a1:
    returnValue = tf.nn.relu(a)
    print(f'{a} after relu will be {returnValue}')
print(tf.nn.relu(a1))

a2 = np.arange(-1, 1, 0.1)
print(a2)
a3 = np.linspace(-3, 3, 100)
print(a3)

print(tf.nn.sigmoid(a3))