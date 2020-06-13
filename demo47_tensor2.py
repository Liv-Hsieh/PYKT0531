import tensorflow as tf
import numpy as np

print(tf.__version__)
a = np.array([5, 3, 8])
b = np.array([3, -1, 2])
c = np.add(a, b)
print(c)

d = tf.constant([5, 3, 8])
e = tf.constant([3, -1, 2])
f = tf.add(d, e)
print(f)
print(f.numpy())
