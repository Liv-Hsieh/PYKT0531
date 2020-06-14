import numpy
from keras import Sequential, callbacks
from keras.layers import Dense
from sklearn.model_selection import train_test_split

dataset1 = numpy.loadtxt("data\\diabetes.csv", delimiter=',', skiprows=1)
print(dataset1.shape)

inputList = dataset1[:, 0:8]  # X
resultList = dataset1[:, 8]  # Y
feature_train, feature_test, label_train, label_test = train_test_split(inputList, resultList,
                                                                        stratify=resultList,
                                                                        test_size=0.2)

classes, counts = numpy.unique(resultList, return_counts=True)
# print(zip(classes, counts))
for cl, co in zip(classes, counts):
    print(f"{int(cl)}==>{co / sum(counts)}")
classes2, counts2 = numpy.unique(label_train, return_counts=True)
for cl, co in zip(classes2, counts2):
    print(f"training: {int(cl)}==>{co / sum(counts2)}")

classes3, counts3 = numpy.unique(label_test, return_counts=True)
for cl, co in zip(classes3, counts3):
    print(f"training: {int(cl)}==>{co / sum(counts3)}")


model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
MODEL_PATH = 'data\\demo55_weight'
# generate a callback


model.fit(inputList, resultList, epochs=200, batch_size=20, validation_data=(feature_test, label_test))
# evaluate
scores = model.evaluate(feature_test, label_test)
print(type(model.metrics_names))
print(type(model.metrics_names))

print(f"{model.metrics_names[1]} value={scores[1] * 100}")
print(f"{model.metrics_names[0]} value={scores[0]}")