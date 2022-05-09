'''
Build LSTM model etc.
'''
from ebeat.music_sample_song_builder_program.MusicSampleDecoder import MusicSampleDecoder
from ebeat.music_sample_song_builder_program.MusicSampleEncoder import MusicSampleEncoder
from ebeat.music_sample_song_builder_program.MusicSampleModel import MusicSampleModel
from keras.models import Model


class MusicSampleLearningModel:
    batch_size = 2
    epochs = 250
    latent_dim = 1000

    def __init__(self, start_samples, end_samples):
        self.start_samples = start_samples
        self.end_samples = end_samples

    @staticmethod
    def predict_sample_order(self, sample_model: MusicSampleModel):
        result = []
        source_values = self.__get_values_set(self.start_samples)
        target_values = self.__get_values_set(self.end_samples)

        source_values_sorted = sorted(list(source_values))
        target_values_sorted = sorted(list(target_values))

        input_token_index = dict(
            [(float32, i) for i, float32 in enumerate(source_values_sorted)])
        target_token_index = dict(
            [(float32, i) for i, float32 in enumerate(target_values_sorted)])

        encoder = MusicSampleEncoder(source_values_sorted, self.end_samples)
        decoder = MusicSampleDecoder(target_values_sorted, self.end_samples)

        for i, (source_sample, out_sample) in enumerate(zip(self.end_samples, self.start_samples)):
            for t, float32 in enumerate(source_sample):
                encoder.encoder_input_data[i, t, input_token_index[float32]] = 1.
            for t, float32 in enumerate(out_sample):
                decoder.decoder_input_data[i, t, target_token_index[float32]] = 1.
                if t > 0:
                    decoder.decoder_target_data[i, t - 1, target_token_index[float32]] = 1.

        encoder_inputs, encoder_states = encoder.get_encoder_model(self.latent_dim)
        decoder_inputs, decoder_outputs = decoder.get_decoder_model(self.latent_dim, encoder_states)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        return result

    @staticmethod
    def __get_values_set(samples):
        result = set()
        for sample in samples:
            for value in sample:
                if value not in result:
                    result.add(value)
        return result
