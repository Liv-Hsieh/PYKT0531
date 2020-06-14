import numpy
from keras import Sequential, callbacks
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier  # for classes(classification)
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

dataset1 = numpy.loadtxt("data\\diabetes.csv", delimiter=',', skiprows=1)
print(dataset1.shape)

inputList = dataset1[:, 0:8]  # X
resultList = dataset1[:, 8]  # Y


def create_default_mode(optimizer='adam', init='uniform'):
    model = Sequential()
    model.add(Dense(10, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_default_mode, verbose=0)
# , epochs=200, batch_size=20
optimizers = ['rmsprop', 'adam', 'sgd']
inits = ['normal', 'uniform']
epochs = [50, 100, 150, 200]
batches = [5, 10, 15]
parameter_grid = dict(optimizer=optimizers, init=inits, epochs=epochs, batch_size=batches)
# parameter_grid = dict(optimizer=['adam'], init=['uniform'], epochs=[200], batch_size=[10])
gridSearchCV = GridSearchCV(estimator=model, param_grid=parameter_grid, cv=3)
grid_result = gridSearchCV.fit(inputList, resultList)

print(f"best:{grid_result.best_score_}, using parameter:{grid_result.best_params_}")