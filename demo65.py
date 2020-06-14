from keras import Sequential
from keras.datasets import imdb
import os
import numpy as np
import matplotlib.pyplot as plt

# copy npz to project\keras_data\imdb.npz
from keras.layers import Dense

(X_train, y_train), (X_test, y_test) = imdb.load_data(path=os.getcwd() + '\\keras_data\\imdb.npz',
                                                      num_words=10000)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(f"max word index={max([max(sequence) for sequence in X_train])}")
# get decoded information
word_index = imdb.get_word_index(path=os.getcwd() + '\\keras_data\\imdb_word_index.json')
print(type(word_index))
reverse_word_index = dict([(v, k) for k, v in word_index.items()])
decoded_review_1 = ' '.join(reverse_word_index.get(i - 3, '?') for i in X_train[0])
print(decoded_review_1)


def vectorize_sequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


X_train = vectorize_sequence(X_train)
X_test = vectorize_sequence(X_test)
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
print(X_train[0])
print(y_train[:5])

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10000,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, epochs=50, batch_size=500,validation_data=(X_test, y_test))