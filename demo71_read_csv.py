import pandas as pd
import keras
from keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from keras import callbacks, Sequential

csv1 = pd.read_csv('data\\bmi.csv')
print(csv1.columns)
csv1['height'] = csv1['height'] / 200
csv1['weight'] = csv1['weight'] / 100

encoder = LabelBinarizer()
transformedLabel = encoder.fit_transform(csv1['label'])
print(csv1['label'][:10])
print(transformedLabel[:10])

test_csv = csv1[25000:]
test_pat = test_csv[['weight', 'height']]
test_ans = transformedLabel[25000:]
train_csv = csv1[:25000]
train_pat = train_csv[['weight', 'height']]
train_ans = transformedLabel[:25000]

model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(2,)))
model.add(Dense(3, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

tbCallback = callbacks.TensorBoard(log_dir="c:\\temp_phw", histogram_freq=1)
history = model.fit(train_pat, train_ans, batch_size=50, epochs=50, verbose=1,
                    validation_data=(test_pat, test_ans), callbacks=[tbCallback])

score = model.evaluate(test_pat, test_ans, verbose=0)
print(f"accuracy={score[1]}, loss={score[0]}")
