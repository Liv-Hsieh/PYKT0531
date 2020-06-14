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

model.fit(inputList, resultList, epochs=200, batch_size=20, validation_split=0.25)
# evaluate
scores = model.evaluate(inputList, resultList)
print(type(model.metrics_names))
print(type(model.metrics_names))

print(f"{model.metrics_names[1]} value={scores[1] * 100}")
print(f"{model.metrics_names[0]} value={scores[0]}")

