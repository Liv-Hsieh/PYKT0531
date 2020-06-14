import keras.utils  as utils

origs = [0, 1, 4, 7, 10]
NUM_DIGITS = 20

for orig in origs:
    converted = utils.to_categorical(orig, NUM_DIGITS)
    print(f"{orig} bocomes {converted}")
