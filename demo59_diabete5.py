import numpy
from keras import Sequential, callbacks
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score

dataset1 = numpy.loadtxt("data\\diabetes.csv", delimiter=',', skiprows=1)
print(dataset1.shape)

inputList = dataset1[:, 0:8]  # X
resultList = dataset1[:, 8]  # Y

def create_default_mode():
    model = Sequential()
    model.add(Dense(10, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model = KerasClassifier(build_fn=create_default_mode, epochs=200, batch_size=20, verbose=0)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model, inputList, resultList, cv=fiveFold)
print(f"performance mean={results.mean()}, std={results.std()}")
