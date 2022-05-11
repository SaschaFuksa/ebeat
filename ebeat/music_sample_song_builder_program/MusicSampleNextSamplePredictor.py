'''
Predicts next best sample with similarity score
'''
from ebeat.music_sample_song_builder_program.MusicSampleConfiguration import MusicSampleConfiguration
from ebeat.music_sample_song_builder_program.MusicSampleLearningModel import MusicSampleLearningModel
from ebeat.music_sample_song_builder_program.MusicSampleLoader import MusicSampleLoader
from ebeat.music_sample_song_builder_program.MusicSampleSongBuilder import MusicSampleSongBuilder


class MusicSampleNextSamplePredictor:

    def __init__(self):
        pass

    @staticmethod
    def create_new_music_file():
        end_samples, start_samples, sample_model = MusicSampleLoader.load_samples()
        learning_model = MusicSampleLearningModel()
        sample_order = learning_model.predict_sample_order(end_samples, start_samples, sample_model)
        MusicSampleSongBuilder.save_song(sample_order)
