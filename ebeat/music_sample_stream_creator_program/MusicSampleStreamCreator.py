import numpy as np

from ebeat.music_sample_stream_creator_program.MusicSampleConfiguration import MusicSampleConfiguration
from ebeat.music_sample_stream_creator_program.MusicSampleDataPreparer import MusicSampleDataPreparer
from ebeat.music_sample_stream_creator_program.MusicSampleLoader import MusicSampleLoader
from ebeat.music_sample_stream_creator_program.MusicSampleModel import MusicSampleModel
from ebeat.music_sample_stream_creator_program.MusicSampleStreamBuilder import MusicSampleStreamBuilder
from ebeat.music_sample_stream_creator_program.MusicSampleStreamPredictor import MusicSampleStreamPredictor


class MusicSampleStreamCreator:

    def __init__(self):
        pass

    @staticmethod
    def create_sample_stream():
        """
        Create train data, model, predict new song structure and save song
        """
        samples, samples_sec_canal = MusicSampleLoader.load_training_samples_fixed_resample_rate(
            MusicSampleConfiguration.train_sample_path)
        edge_size = min(map(len, samples))
        x_train, y_train, x_val, y_val = MusicSampleDataPreparer.prepare_data(samples, samples_sec_canal, edge_size)
        x_train = np.array(x_train)
        x_train = x_train.reshape(-1, 2 * edge_size)
        y_train = np.array(y_train)
        y_train = y_train.reshape(-1, 1)
        model = MusicSampleModel.create_model(edge_size)
        if MusicSampleConfiguration.use_model:
            model.load_weights(MusicSampleConfiguration.model_path)
        else:
            model.fit(x_train, y_train, epochs=750, batch_size=64)
        selected_samples = MusicSampleStreamPredictor.predict_classification_stream(model)
        MusicSampleStreamBuilder.save_song(selected_samples)
