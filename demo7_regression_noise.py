import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

data1 = datasets.make_regression(100, 1, noise=5)
print(type(data1), data1[0].shape, data1[1].shape)
plt.scatter(data1[0], data1[1], c='red', marker='.')
plt.show()

regression1 = linear_model.LinearRegression()
regression1.fit(data1[0], data1[1])
print(f"coef={regression1.coef_[0]}, intercept={regression1.intercept_}")
print(f"score={regression1.score(data1[0], data1[1])}")

range1 = [data1[0].min() - 1, data1[0].max() + 1]
plt.plot(range1,regression1.coef_ * range1 + regression1.intercept_)
plt.scatter(data1[0], data1[1], c='red', marker='.')
plt.show()
