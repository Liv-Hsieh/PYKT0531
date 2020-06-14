import pandas as pd 
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

df1 = pd.read_csv("data\\iris.data", header=None)
dataset = df1.values
print(type(df1), type(dataset))
print(type(dataset[0][0]))

features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]
# print(features[:5])
# print(features.shape)
# print(labels.shape)
# print(np.unique(labels, return_counts=True))
# print(labels)

# alphabetic ==> digit (index)
encoder = LabelEncoder().fit(labels)
encoded_Y = encoder.transform(labels)
print(type(encoded_Y))
print(np.unique(encoded_Y, return_counts=True))
print(encoded_Y[:5])
# convert to 1-hot encoding
dummy_y = np_utils.to_categorical(encoded_Y)
print(type(dummy_y), dummy_y.shape)
print(np.unique(dummy_y, return_counts=True))
print(dummy_y[:5])

def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model,
                            epochs=200, batch_size=30, verbose=1)
kfold = KFold(n_splits=3, shuffle=True)
results = cross_val_score(estimator, features, dummy_y, cv=kfold)
print(f"accuracy={results.mean()}, std={results.std()}")