'''
Predicts next best sample with similarity score
'''
from ebeat.music_sample_song_builder_program.MusicSampleLearningModel import MusicSampleLearningModel
from ebeat.music_sample_song_builder_program.MusicSampleLoader import MusicSampleLoader
from ebeat.music_sample_song_builder_program.MusicSampleSongBuilder import MusicSampleSongBuilder


class MusicSampleNextSamplePredictor():

    def __init__(self, input_directory: str, output_directory: str, model_path: str):
        self.model_path = model_path
        self.output_directory = output_directory
        self.input_directory = input_directory

    def create_new_music_file(self):
        loader = MusicSampleLoader(self.input_directory)
        end_samples, start_samples, sample_model = loader.load_samples()
        learning_model = MusicSampleLearningModel(end_samples, start_samples)
        sample_order = learning_model.predict_sample_order(sample_model)
        song_builder = MusicSampleSongBuilder(self.output_directory)
        song_builder.save_song(sample_order)

