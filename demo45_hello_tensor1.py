import tensorflow as tf

print(tf.__version__)
h1 = tf.constant("Hello Tensorflow")
print(type(h1))
print(h1.numpy())