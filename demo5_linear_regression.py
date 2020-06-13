import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

features = [[-1], [1], [2], [3], [4]]
values = [-10, 1, 4, 5, 20]
plt.scatter(features, values, c='green')
plt.show()

regression1 = linear_model.LinearRegression()
regression1.fit(features, values)
print(f'coef={regression1.coef_}')
print(f'intercept={regression1.intercept_}')
# range1 = [-4, 4]
xrange = np.array(features)
range1 = [xrange.min(), xrange.max()]
print(np.array(range1).reshape(-1, 1))
plt.plot(range1, regression1.predict(np.array(range1).reshape(-1,1)), c='blue',
         label=f'y={regression1.coef_[0]:.2f}x+{regression1.intercept_:.2f}')

#plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_, c='red',
#         label=f'y={regression1.coef_[0]:.2f}x+{regression1.intercept_:.2f}')
plt.scatter(features, values, c='red')
plt.legend()
plt.show()

y = regression1.predict(features)
print(y)
y_hat = values
print(((y-y_hat)**2).mean()) # mean square of error
