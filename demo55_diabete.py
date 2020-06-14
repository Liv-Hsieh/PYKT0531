import numpy
from keras import Sequential, callbacks
from keras.layers import Dense

dataset1 = numpy.loadtxt("data\\diabetes.csv", delimiter=',', skiprows=1)
print(dataset1.shape)

inputList = dataset1[:, 0:8]  # X
resultList = dataset1[:, 8]  # Y

model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
MODEL_PATH = 'data\\demo55_weight'
# generate a callback
mc_callback = callbacks.ModelCheckpoint(filepath=MODEL_PATH, save_weights_only=True, verbose=1)

model.fit(inputList, resultList, epochs=200, batch_size=20, callbacks=[mc_callback])
# evaluate
scores = model.evaluate(inputList, resultList)
print(type(model.metrics_names))
print(type(model.metrics_names))

print(f"{model.metrics_names[1]} value={scores[1] * 100}")
print(f"{model.metrics_names[0]} value={scores[0]}")

## create second model
model2 = Sequential()
model2.add(Dense(10, input_dim=8, activation='relu'))
model2.add(Dense(8, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
print(model2.summary())
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

scores2 = model2.evaluate(inputList, resultList)

print(f"before weight: {model2.metrics_names[1]} value={scores2[1] * 100}")
print(f"before weight: {model2.metrics_names[0]} value={scores2[0]}")
model2.load_weights(MODEL_PATH)

scores2 = model2.evaluate(inputList, resultList)
print(f"after load weight: {model2.metrics_names[1]} value={scores2[1] * 100}")
print(f"after load weight: {model2.metrics_names[0]} value={scores2[0]}")