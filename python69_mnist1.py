import tensorflow as tf
import numpy as np
import keras.utils as utils
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

FLATTEN_DIM = 28 * 28
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

trainImages = np.reshape(train_images, (TRAINING_SIZE, FLATTEN_DIM))
testImages = np.reshape(test_images, (TEST_SIZE, FLATTEN_DIM))
print(type(trainImages[0]), type(trainImages[0][0]))
# convert uint to float
trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)
print(type(trainImages[0][0]))
trainImages /= 255
testImages /= 255
NUM_DIGITS = 10

trainLabels = utils.to_categorical(train_labels, NUM_DIGITS)
testLabels = utils.to_categorical(test_labels, NUM_DIGITS)

model = Sequential()
model.add(Dense(units=128, activation=tf.nn.relu, input_shape=(FLATTEN_DIM,)))
model.add(Dense(units=10, activation=tf.nn.softmax))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tbCallback = TensorBoard(log_dir="c:\\temp_phw", histogram_freq=0, write_graph=True, write_images=True)

model.fit(trainImages, trainLabels, epochs=10, callbacks=[tbCallback])

predictLabels = model.predict_classes(testImages)
print("result=", predictLabels[:10])

loss, accuracy = model.evaluate(testImages, testLabels)
print(f"test accuracy={accuracy}, loss value={loss}")
