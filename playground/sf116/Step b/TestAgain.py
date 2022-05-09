import difflib
import os

from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from pydub import AudioSegment
from tensorflow import keras

import SamplingConstants
from SampleModel import SampleModel

EDGE_SIZE = 70


def load_sample_edges(path: str):
    start_samples = []
    end_samples = []
    files = os.listdir(path)
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split(".")[0]))
    print(files[:11])
    i = 0
    sampleModel = []
    for file_name in files:
        if 'wav' in file_name:
            complete_path = path + file_name
            sample = AudioSegment.from_wav(complete_path)
            sample_array = sample.get_array_of_samples()
            start_array = sample_array[:EDGE_SIZE]
            start_samples.append(start_array)
            end_array = sample_array[-EDGE_SIZE:]
            end_samples.append(end_array)
            model = SampleModel(name=file_name, start=start_array, end=end_array)
            sampleModel.append(model)
            i += 1

    end_samples = end_samples[:-1]
    start_samples = start_samples[1:]

    return end_samples, start_samples, sampleModel


end_samples, start_samples, sampleModel = load_sample_edges(SamplingConstants.IN_DIRECTORY)
'''print(type(end_samples))

start_samples_tensor = tensorflow.convert_to_tensor(start_samples)
end_samples_tensor = tensorflow.convert_to_tensor(end_samples)

start_samples_tensor = tensorflow.reshape(start_samples_tensor, shape=(10, 1900, 1))
end_samples_tensor = tensorflow.reshape(end_samples_tensor, shape=(10, 1900, 1))

model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(None, 1)))  # (timesteps, features)
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse',
              metrics=['accuracy'], )
model.summary()

model.fit(end_samples_tensor, start_samples_tensor, epochs=10, batch_size=10, verbose=2)

x_input = tensorflow.convert_to_tensor(end_samples[2])
x_input = tensorflow.reshape(x_input, shape=(10, 100, 1))
print(model.predict(x_input, verbose=2))

print('end sample: ' + str(start_samples_tensor[2]))'''

### Beispiel aus anderem projekt:

import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import Model

batch_size = 2  # Batch size for training.
epochs = 250  # Number of epochs to train for.
latent_dim = 1000  # Latent dimensionality of the encoding space.
# num_samples = 10000  # Number of samples to train on.

source_samples = []
target_samples = []
source_values = set()
target_values = set()

# in_samples = [[2.1, 4.6, 5.0], [3.7, 4.9, 1.0], [3.5, 4.0, 5.7], [0.3, 8.5, 0.4], [2.1, 3.3, 0.9], [3.7, 4.0, 2.7]]
# out_samples = [[3.7, 4.9, 1.0], [3.5, 4.0, 5.7], [0.3, 8.5, 0.4], [2.1, 3.3, 0.9], [3.7, 4.0, 2.7], [7.0, 0.8, 0.9]]

i = 0

while i < 10:
    print(end_samples[i])
    print(start_samples[i])
    i += 1

# sys.exit()

i = 0

while i < len(end_samples):
    source_samples.append(end_samples[i])
    target_samples.append(start_samples[i])
    for value in end_samples[i]:
        if value not in source_values:
            source_values.add(value)
    for value in start_samples[i]:
        if value not in target_values:
            target_values.add(value)
    i += 1

source_values_sorted = sorted(list(source_values))
target_values_sorted = sorted(list(target_values))
num_encoder_tokens = len(source_values_sorted)
num_decoder_tokens = len(target_values_sorted)
max_encoder_seq_length = max([len(txt) for txt in source_samples])  # 3
max_decoder_seq_length = max([len(txt) for txt in target_samples])  # 3

print('Number of samples:', len(source_samples))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(float32, i) for i, float32 in enumerate(source_values_sorted)])
print('input_token_index: ' + str(input_token_index))
target_token_index = dict(
    [(float32, i) for i, float32 in enumerate(target_values_sorted)])
print('target_token_index: ' + str(target_token_index))

encoder_input_data = np.zeros(
    (len(source_samples), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(source_samples), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(source_samples), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (source_sample, out_sample) in enumerate(zip(source_samples, start_samples)):
    for t, float32 in enumerate(source_sample):
        encoder_input_data[i, t, input_token_index[float32]] = 1.
    for t, float32 in enumerate(out_sample):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[float32]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[float32]] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
filepath = "model/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint]
'''model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=callbacks_list)'''
# Save model
#model.save('model/ebeat_model-new.h5')
#model.save_weights('model/ebeat_weights-new.h5')

#model = keras.models.load_model('model/weights-improvement-215-0.0084-bigger.hdf5')
model.load_weights('model/weights-improvement-215-0.0084-bigger.hdf5')

# model = tensorflow.keras.models.load_model('s2s_2_2.h5')

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, float32) for float32, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, float32) for float32, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    # target_seq[0, 0, target_token_index[1.]] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if len(decoded_sentence) >= max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in [0, 53]:
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Start sentence:', source_samples[seq_index])
    print('End sentence:', target_samples[seq_index])
    print('Decoded sentence:', decoded_sentence)

    plt.plot(target_samples[seq_index])  # plotting by columns
    plt.show()
    plt.plot(decoded_sentence)  # plotting by columns
    plt.show()

max_length = len(source_samples) - 1

already_used = set()
already_used.add(sampleModel[0].name)
already_used.add(sampleModel[53].name)


def predict_next_sample(index, actual):
    input_seq = encoder_input_data[index: index + 1]
    decoded_sentence = decode_sequence(input_seq)
    ratio = 0.0
    name = ''
    next_end = []
    for sample in sampleModel:
        sm = difflib.SequenceMatcher(None, sample.start.tolist(), decoded_sentence)
        new_ratio = sm.ratio()
        if (new_ratio > ratio) and (sample.name not in already_used):
            ratio = new_ratio
            name = sample.name
            next_end = sample.end
    print('predicted sample: ' + name + ' with ratio ' + str(ratio))
    already_used.add(name)
    actual += 1
    if ratio is 0.0:
        print('No further sample found after ' + str(len(already_used)) + ' samples.')
    elif actual < max_length:
        new_index = end_samples.index(next_end)
        predict_next_sample(new_index, actual)


print('start with : ' + sampleModel[0].name)
predict_next_sample(1, 0)

# 0.0174
