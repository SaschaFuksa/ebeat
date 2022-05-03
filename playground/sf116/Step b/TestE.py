import os

import tensorflow as tf
from pydub import AudioSegment

import SamplingConstants

training = os.listdir(SamplingConstants.IN_DIRECTORY)
training_tensors = []
for sample_name in training:
    if 'wav' in sample_name:
        file_path = os.path.join(SamplingConstants.IN_DIRECTORY, sample_name)
        # test_file = tf.io.read_file(file_path)
        # test_audio, _ = tf.audio.decode_wav(contents=test_file)
        # training_tensors.append(test_audio)
        song = AudioSegment.from_wav(file_path)
        samples = song.get_array_of_samples()
        training_tensors.append(samples[:100])

# full_sample = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
# start = full_sample[:3]
# ende = full_sample[-3:]
# [start, ende] -> [[10, 10, 10], [10, 10, 10]]

# training = tf.concat(training_tensors, axis=0)

# validation_tensors = training_tensors[1:]

# validation = tf.concat(validation_tensors, axis=0)

single_file = os.path.join(SamplingConstants.IN_SINGLE_SAMPLE_DIRECTORY, 'Tea K Pea - nauticals_12.wav')

model = tf.keras.Sequential([tf.keras.layers.Dense(units=4, input_shape=[2]), tf.keras.layers.Dense(units=2)])
# model = tf.keras.Sequential([tf.keras.layers.Dense(units=4, input_shape=[440100, 2]), tf.keras.layers.Dense(units=2)])
# Create CNN model
'''model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
model.add(tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='sigmoid'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='sigmoid'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='sigmoid'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='softmax'))'''

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=['accuracy'],
)

# Train model for 10 epochs, capture the history
history = model.fit(training_tensors[:10], epochs=2, validation_data=training_tensors[1:10])

print("Hallo: " + model.predict(single_file))
