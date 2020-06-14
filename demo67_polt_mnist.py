import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

print(f'train image shape={train_images.shape}, test image shape={test_images.shape}')
print(f'train label count={len(train_labels)}, test label count={len(test_labels)}')

def plotImage(index):
    plt.title(f'Train image marked as {train_labels[index]}')
    plt.imshow(train_images[index], cmap='binary')
    plt.show()

def plotTestImage(index):
    plt.title(f'Train image marked as {test_labels[index]}')
    plt.imshow(test_images[index], cmap='binary')
    plt.show()

plotImage(0)