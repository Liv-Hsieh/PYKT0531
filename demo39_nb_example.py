import numpy as np
from sklearn.naive_bayes import GaussianNB

x = np.array([[-1, -1], [-2, -1], [-3, -2],
              [1, 1], [2, 1], [3, 2]])

Y = np.array([1, 1, 1, 3, 2, 2])
nb1 = GaussianNB()
nb1.fit(x, Y)
points = [[-0.8, -0.8], [0.8, -0.8], [-0.8, 0.8], [-0.8, -0.8]]
print(f"print result={nb1.predict(points)}")

nb2 = GaussianNB()
nb2.partial_fit(x, Y, np.unique(Y))
print(f"[nb2]print result={nb2.predict(points)}")
nb2.partial_fit([[1,-1]],[3])
print(f"[nb2]print result={nb2.predict(points)}")