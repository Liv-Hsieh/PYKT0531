import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
print(tf.__version__)
a = np.array([3, 4, 5])
b = np.array([6, 7, 8])
c = np.add(a, b)
print(c)

d = tf.constant([3, 4, 5])
e = tf.constant([6, 7, 8])
f = tf.add(d, e)
print(f)
s1 = tf.compat.v1.Session()
print(s1.run(f))