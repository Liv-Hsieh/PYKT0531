from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x, y = make_regression(n_samples=100,n_features=10,n_informative=5)
print(x.shape)
model = LinearRegression()
model.fit(x,y)