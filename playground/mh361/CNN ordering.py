import os
import tensorflow as tf
from pydub import AudioSegment
import Paths
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
import torch

training = os.listdir(Paths.IN_DIRECTORY)
training_tensors = []
# tensor = torch.tensor([])
for sample_name in training:
    if 'wav' in sample_name:
        file_path = os.path.join(Paths.IN_DIRECTORY, sample_name)
        # test_file = tf.io.read_file(file_path)
        # test_audio, _ = tf.audio.decode_wav(contents=test_file)
        # training_tensors.append(test_audio)
        song = AudioSegment.from_wav(file_path)
        samples = song.get_array_of_samples()
        training_tensors.append(samples[:100])
#print(training_tensors)
#tensor = torch.FloatTensor(training_tensors)
Input_tensor = tf.convert_to_tensor(training_tensors, dtype=float, dtype_hint=None, name=None)
print(training_tensors)

'''
print(Input_tensor[1:])
nRows, nCols, nDims = Input_tensor[1:]
train_data = Input_tensor.reshape(Input_tensor.shape[0], nRows, nCols, nDims)
test_data = Input_tensor.reshape(Input_tensor.shape[0], nRows, nCols, nDims)
input_shape = (nRows, nCols, nDims)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')


def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)



def createModel():
    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))


    return model




single_file = os.path.join(Paths.IN_SINGLE_SAMPLE_DIRECTORY, 'Tea K Pea - nauticals_12.wav')

# Create CNN model
model = tf.keras.Sequential([tf.keras.layers.Dense(units=4, input_shape=[2]), tf.keras.layers.Dense(units=2)])


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

# Train model for 5 epochs, capture the history
history = model.fit(training_tensors[:10], epochs=2, validation_data=training_tensors[1:10])

#model.predict()


'''