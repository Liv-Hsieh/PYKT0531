import tensorflow as tf

x1 = tf.ones(1)
x2 = tf.zeros(1)
print(x1, x2)
x3 = tf.zeros([1, 5])
print(x3)
x4 = tf.ones([5, 1])
print(x4)
x5 = tf.ones([2, 2])
y1 = tf.matmul(x5, x5)
print(y1.numpy())
x6 = tf.ones([2, 4])
x7 = tf.ones([4, 2])
y2 = tf.matmul(x6, x7)
print(y2.numpy())
# y3 = tf.matmul(x7, x6)
# print(y3.numpy())
# x8 = tf.constant([[1, 2, 3]])
# x9 = tf.constant([[4], [5], [6]])
# y4 = tf.matmul(x8, x9)
# print(y4)
# y5 = tf.matmul(x9, x8)
# print(y5)