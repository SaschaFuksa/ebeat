'''
Wrapps all encoding model code
'''
import numpy
from keras import Input
from keras.layers import LSTM

from ebeat.music_sample_song_builder_program.MusicSampleConfiguration import MusicSampleConfiguration


class MusicSampleEncoderModel:

    def __init__(self, source_values_sorted, end_samples):
        self.num_encoder_tokens = len(source_values_sorted)
        self.max_encoder_seq_length = max([len(txt) for txt in end_samples])
        self.encoder_input_data = numpy.zeros(
            (len(end_samples), self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')

    def get_encoder_data(self):
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(MusicSampleConfiguration.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        return encoder_inputs, encoder_states
