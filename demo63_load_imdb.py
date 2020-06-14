from keras.datasets import imdb
import os
import numpy as np
import matplotlib.pyplot as plt

# copy npz to project\keras_data\imdb.npz
(X_train, y_train), (X_test, y_test) = imdb.load_data(path=os.getcwd() + '\\keras_data\\imdb.npz')
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
X = np.concatenate((X_train, X_test), axis=0)
Y = np.concatenate((y_train, y_test), axis=0)
print(X.shape)
print(Y.shape)
print(X[0])
print(np.unique(Y, return_counts=True))

print(f'total {len(np.unique(np.hstack(X)))} distinct words in IMDB')
result = [len(x) for x in X]
print(f"every review length={result[:10]}")
print(f"review length={np.mean(result)}, std={np.std(result)}")

plt.subplot(121)
plt.boxplot(result)
plt.subplot(122)
plt.hist(result)
plt.show()