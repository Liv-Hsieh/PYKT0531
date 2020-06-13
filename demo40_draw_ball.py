import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

X = np.array([[-1, -1], [-2, -1], [-3, -2],
              [1, 1], [2, 1], [3, 2]])

Y = np.array([1, 1, 2, 1, 1, 2])

X_MIN, X_MAX = -4, 4
Y_MIN, Y_MAX = -4, 4

h = .005
XX, YY = np.meshgrid(np.arange(X_MIN, X_MAX, h), np.arange(Y_MIN, Y_MAX, h))
classifier = GaussianNB()
classifier.fit(X, Y)
Z = classifier.predict(np.c_[XX.ravel(), YY.ravel()])

Z = Z.reshape(XX.shape)
plt.xlim(X_MIN, X_MAX)
plt.ylim(Y_MIN, Y_MAX)
plt.pcolormesh(XX, YY, Z)

XB = []
YB = []
XR = []
YR = []
index = 0
for index in range(0, len(Y)):
    if Y[index] == 1:
        print(f"B equal to {X[index, :]}")
        XB.append(X[index, 0])
        YB.append(X[index, 1])
    if Y[index] == 2:
        print(f"R equal to {X[index, :]}")
        XR.append(X[index, 0])
        YR.append(X[index, 1])

plt.scatter(XB, YB, color='b', label='Blue')
plt.scatter(XR, YR, color='r', label='RED')
plt.legend()
plt.show()
