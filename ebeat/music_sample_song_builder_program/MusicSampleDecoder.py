'''
Wrapps all decoding model code
'''

import numpy
from keras import Input
from keras.layers import LSTM, Dense


class MusicSampleDecoder:

    def __init__(self, target_values_sorted, end_samples):
        self.num_decoder_tokens = len(target_values_sorted)
        self.max_decoder_seq_length = max([len(txt) for txt in end_samples])

        self.decoder_input_data = numpy.zeros(
            (len(end_samples), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')
        self.decoder_target_data = numpy.zeros(
            (len(end_samples), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')

    def get_decoder_model(self, latent_dim: int, encoder_states):
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        return decoder_inputs, decoder_outputs