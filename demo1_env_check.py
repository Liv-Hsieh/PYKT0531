import tensorflow
import numpy
import pandas

print(F'tensorflow version = {tensorflow.__version__}')
print(F'numpy version = {numpy.__version__}')
print(F'pandas version = {pandas.__version__}')

x1 = [1, 2, 3]
x2 = [4, 5, 6]
print(type(x1), type(x2))
print(F'x1 + x2 = {x1 + x2}')
y1 = numpy.array(x1)
y2 = numpy.array(x2)
print(type(y1), type(y2))
print(F'y1 + y2 = {y1 + y2}')
z1 = tensorflow.constant(x1)
z2 = tensorflow.constant(x2)
print(type(z1), type(z2))
print(F'z1 + z2 = {z1 + z2}')
print(F'z1 + z2 use tensor = {tensorflow.add(z1, z2)}')
print(z1)
print(z1.numpy())