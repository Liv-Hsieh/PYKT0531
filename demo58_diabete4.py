import numpy
from keras import Sequential, callbacks
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold

dataset1 = numpy.loadtxt("data\\diabetes.csv", delimiter=',', skiprows=1)
print(dataset1.shape)

inputList = dataset1[:, 0:8]  # X
resultList = dataset1[:, 8]  # Y

# initial StratifiedKFold
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
totalScores = []


for train, test in fiveFold.split(inputList, resultList):
    model = Sequential()
    model.add(Dense(10, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    data = [resultList[train], resultList[test]]
    for d in data:
        classes, counts = numpy.unique(d, return_counts=True)
        for cl, co in zip(classes, counts):
            print(f"{cl} ratio={co / sum(counts)}")
    model.fit(inputList[train], resultList[train], epochs=200, batch_size=20, verbose=0)
    # evaluate
    scores = model.evaluate(inputList[test], resultList[test])

    print(f"{model.metrics_names[1]} value={scores[1] * 100}")
    print(f"{model.metrics_names[0]} value={scores[0]}")
    totalScores.append(scores[1] * 100)

print(f"total scores = {totalScores}, avg={numpy.mean(totalScores)}, std={numpy.std(totalScores)}")