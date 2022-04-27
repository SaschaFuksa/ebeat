import tensorflow as tf
from keras import models, layers, Input

import SamplingConstants

test_file = tf.io.read_file(SamplingConstants.IN_SINGLE_SAMPLE_DIRECTORY + SamplingConstants.IN_SINGLE_FILE_NAME)
test_audio, _ = tf.audio.decode_wav(contents=test_file)

test_audio = tf.expand_dims(test_audio, axis=2)

model = models.Sequential()
model.add(Input(tensor=test_audio))
model.add(layers.Conv1D(filters=2, kernel_size=2, activation='sigmoid', padding='valid', input_shape=(441000, 2)))
