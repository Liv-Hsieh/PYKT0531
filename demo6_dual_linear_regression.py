import matplotlib.pyplot as plt
from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 8], [3, 10], [6, 6]]
values = [1, 4, 5.5, 8, 10]
regression1 = linear_model.LinearRegression()
regression1.fit(features, values)
print(regression1.coef_)
print(regression1.intercept_)
points = [[0.8, 0.8], [2, 1], [5, 7], [7, 5]]
result = regression1.predict(points)
print(f"points:{points}, result={result}")
print(f"depend on input data, regression1 score={regression1.score(features, values)}")
print(f"depend on other data, regression1 score={regression1.score(features, [2, 5, 8, 10, 18])}")
print(f"depend on input data, regression1 score={regression1.score(points, result)}")
