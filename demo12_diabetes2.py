import numpy as np
from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
Y = diabetes.target
print(type(diabetes), type(X), type(Y))
print(X.shape)
print(Y.mean(), Y.std())

dataForTest = -10
data_train = X[:dataForTest]
target_train = Y[:dataForTest]
print(data_train.shape)
print(target_train.shape)

data_test = X[dataForTest:]
target_test = Y[dataForTest:]
print(data_test.shape)
print(target_test.shape)

# model fit
regression1 = linear_model.LinearRegression().fit(data_train, target_train)
print(f"coef={regression1.coef_}")
print(f"intercept={regression1.intercept_}")

# score
print(f"score for training data:{regression1.score(data_train, target_train)}")
print(f"score for testing data:{regression1.score(data_test, target_test)}")

for i in range(dataForTest, 0):
    # print(data_test[i])
    dataArray = np.array(data_test[i]).reshape(1, -1)
    print(f"predict={regression1.predict(dataArray)[0]}/actual={target_test[i]}")
mse = np.mean(np.square(regression1.predict(data_test) - target_test))
print(f"mean square error={mse}")

s = np.mean(np.square(regression1.predict(data_train))-target_train)
print(s)