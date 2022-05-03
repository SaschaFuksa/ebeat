# Beispiel mit tensor aber ohne Verwendungszweck bisher, da nur 1 tensor und keine Liste an Tensoren
import tensorflow as tf
from keras import models, layers, Input
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

import SamplingConstants

test_file = tf.io.read_file(SamplingConstants.IN_SINGLE_SAMPLE_DIRECTORY + SamplingConstants.IN_SINGLE_FILE_NAME)
test_audio, _ = tf.audio.decode_wav(contents=test_file)

# Another example to expand dims
# test_audio = tf.expand_dims(test_audio, axis=2)
test_audio = test_audio.reshape(-1, 441000, 2)
# https://datascience.stackexchange.com/questions/15056/how-to-use-lists-in-tensorflow
# https://apfalz.github.io/rnn/rnn_demo.htmlhttps://apfalz.github.io/rnn/rnn_demo.html

model = models.Sequential()
model.add(Input(tensor=test_audio))
model.add(layers.Conv1D(filters=2, kernel_size=2, activation='sigmoid', padding='valid', input_shape=(441000, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling1D(pool_size=(2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv1D(64, 3, padding='same', activation='sigmoid'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling1D(pool_size=(2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv1D(128, 3, padding='same', activation='sigmoid'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling1D(pool_size=(2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='softmax'))

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=['accuracy'],
)

# Train model for 10 epochs, capture the history
history = model.fit(test_audio, epochs=3)
