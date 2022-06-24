from ebeat.music_sample_stream_creator_program.MusicSampleConfiguration import MusicSampleConfiguration
from ebeat.music_sample_stream_creator_program.MusicSampleLoader import MusicSampleLoader


class MusicSampleStreamCreator:

    def __init__(self):
        pass

    @staticmethod
    def create_sample_stream():
        train_data = MusicSampleLoader.load_training_samples_fixed_resample_rate(MusicSampleConfiguration.train_sample_path)