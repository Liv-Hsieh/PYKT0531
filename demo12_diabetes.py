import numpy as np
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
Y = diabetes.target
print(type(diabetes), type(X), type(Y))
print(X.shape)
print(Y.mean(), Y.std())

dataForTest = 60
data_train = X[:dataForTest]
target_train = Y[:dataForTest]
print(data_train.shape)
print(target_train.shape)
