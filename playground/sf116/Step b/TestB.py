import os
import wave
from pathlib import Path

import pylab
import tensorflow as tf

import SamplingConstants

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

