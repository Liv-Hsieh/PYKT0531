import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
print(list(iris.keys()))
print(iris.feature_names)
print(iris.target_names)
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)

plt.plot(X, y, 'g.')

regression1 = LogisticRegression().fit(X, y)
print(f"coef={regression1.coef_}, intercept={regression1.intercept_}")

X_new = np.linspace(0, 3, 1000).reshape(-1,1)
y_prob = regression1.predict_proba(X_new)
print(y_prob)
plt.plot(X_new, y_prob[:,1],"r--", label="Virginica")
plt.plot(X_new, y_prob[:,0],'g-', label='not virginica')
plt.legend()
plt.show()