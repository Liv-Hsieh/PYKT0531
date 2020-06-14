import tensorflow as tf

p = tf.random.uniform((10, 10))
# print(p)
q = tf.random.normal((10, 10))
# print(q)
s = tf.random.gamma((10, 10),1)
print(s)
