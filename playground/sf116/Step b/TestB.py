import os
import wave
from pathlib import Path

import pylab
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

import SamplingConstants

np_config.enable_numpy_behavior()

all_samples = os.listdir(SamplingConstants.IN_DIRECTORY)


def create_specs():
    for sample_name in all_samples:
        if 'wav' in sample_name:
            file_path = os.path.join(SamplingConstants.IN_DIRECTORY, sample_name)
            wav = wave.open(file_path, 'r')
            frames = wav.readframes(-1)
            file_stem = Path(file_path).stem
            file_dist_path = os.path.join(SamplingConstants.IN_TRAINING, file_stem)
            sound_info = pylab.frombuffer(frames, 'int16')
            frame_rate = wav.getframerate()
            wav.close()
            pylab.specgram(sound_info, Fs=frame_rate)
            pylab.savefig(f'{file_dist_path}.png')
            pylab.close()


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
BATCH_SIZE = 32

image = tf.keras.preprocessing.image_dataset_from_directory(
    batch_size=BATCH_SIZE,
    directory=SamplingConstants.IN_SINGLE_SAMPLE_DIRECTORY,
    shuffle=True,
    color_mode='rgb',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    seed=0)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    batch_size=BATCH_SIZE,
    directory=SamplingConstants.IN_TRAINING,
    shuffle=True,
    color_mode='rgb',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    seed=0)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    batch_size=BATCH_SIZE,
    directory=SamplingConstants.IN_CLASSIFICATION,
    shuffle=True,
    color_mode='rgb',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    seed=0)

print(validation_dataset)

# Create CNN model
model = tf.keras.models.Sequential()
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
model.add(tf.keras.layers.Dense(4, activation='softmax'))

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=['accuracy'],
)

# Train model for 10 epochs, capture the history
history = model.fit(train_dataset, epochs=2, validation_data=validation_dataset)

print(model.predict(image))
