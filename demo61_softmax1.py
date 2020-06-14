import numpy as np
import tensorflow as tf

scores = np.array([4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])

print(f"ratio = {scores / sum(scores)}")

print(f"softmax ratio = {np.exp(scores) / np.sum(np.exp(scores), axis=0)}")

print(f"softmax by tensorflow nn function={tf.nn.softmax(scores).numpy()}")