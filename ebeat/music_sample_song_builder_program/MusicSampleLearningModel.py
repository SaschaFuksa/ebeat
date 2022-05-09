'''
Build LSTM model etc.
'''
from keras.callbacks import ModelCheckpoint
from keras.models import Model

from ebeat.music_sample_song_builder_program.MusicSampleDecoder import MusicSampleDecoder
from ebeat.music_sample_song_builder_program.MusicSampleDecoderModel import MusicSampleDecoderModel
from ebeat.music_sample_song_builder_program.MusicSampleEncoderModel import MusicSampleEncoderModel
from ebeat.music_sample_song_builder_program.MusicSampleModel import MusicSampleModel
from ebeat.music_sample_song_builder_program.MusicSampleSimilarityPredictor import MusicSampleSimilarityPredictor


class MusicSampleLearningModel:
    batch_size = 2
    epochs = 250
    latent_dim = 1000

    def __init__(self, start_samples, end_samples):
        self.start_samples = start_samples
        self.end_samples = end_samples

    @staticmethod
    def predict_sample_order(self, sample_model: MusicSampleModel):
        source_values = self.__get_values_set(self.start_samples)
        target_values = self.__get_values_set(self.end_samples)

        source_values_sorted = sorted(list(source_values))
        target_values_sorted = sorted(list(target_values))

        input_token_index = dict(
            [(float32, i) for i, float32 in enumerate(source_values_sorted)])
        target_token_index = dict(
            [(float32, i) for i, float32 in enumerate(target_values_sorted)])

        encoder_model_builder = MusicSampleEncoderModel(source_values_sorted, self.end_samples)
        decoder_model_builder = MusicSampleDecoderModel(target_values_sorted, self.end_samples)

        for i, (source_sample, out_sample) in enumerate(zip(self.end_samples, self.start_samples)):
            for t, float32 in enumerate(source_sample):
                encoder_model_builder.encoder_input_data[i, t, input_token_index[float32]] = 1.
            for t, float32 in enumerate(out_sample):
                decoder_model_builder.decoder_input_data[i, t, target_token_index[float32]] = 1.
                if t > 0:
                    decoder_model_builder.decoder_target_data[i, t - 1, target_token_index[float32]] = 1.

        encoder_inputs, encoder_states = encoder_model_builder.get_encoder_data(self.latent_dim)
        decoder_inputs, decoder_outputs = decoder_model_builder.get_decoder_data(self.latent_dim, encoder_states)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        callbacks_list = self.__get_callbacks()
        model.fit([encoder_model_builder.encoder_input_data, decoder_model_builder.decoder_input_data],
                  decoder_model_builder.decoder_target_data,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_split=0.2,
                  callbacks=callbacks_list)

        encoder_model = Model(encoder_inputs, encoder_states)
        decoder_model = decoder_model_builder.build_decoder_model(self.latent_dim, decoder_inputs)

        decoder = MusicSampleDecoder(encoder_model, decoder_model, decoder_model_builder, target_token_index)
        similarity_predictor = MusicSampleSimilarityPredictor(decoder, sample_model, self.end_samples,
                                                              encoder_model_builder.encoder_input_data)
        similarity_predictor.predict_next_samples_recursive(0)
        return similarity_predictor.already_used_samples

    @staticmethod
    def __get_values_set(samples):
        result = set()
        for sample in samples:
            for value in sample:
                if value not in result:
                    result.add(value)
        return result

    @staticmethod
    def __get_callbacks():
        filepath = "model/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        return [checkpoint]
