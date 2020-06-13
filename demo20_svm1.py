import numpy as np
from sklearn.svm import SVC

X = np.array([[-1, -1], [-2, -1], [-3, -3], [1, 1], [2, 1], [3, 3]])
y = np.array([1, 1, 1, 2, 2, 2])

svc1 = SVC()
svc1.fit(X, y)
print(f"predict={svc1.predict([[-0.5, 0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])}")