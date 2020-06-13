import tensorflow as tf

tf.compat.v1.disable_eager_execution()
print(tf.__version__)
hello = tf.constant('Hello Tensorflow (back to legacy)')
print(type(hello))
print(hello)
#print(hello.numpy())
s1 = tf.compat.v1.Session()
print(s1.run(hello))
#print(hello.numpy())