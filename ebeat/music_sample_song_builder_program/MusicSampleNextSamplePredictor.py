'''
Predicts next best sample with similarity score
'''
from ebeat.music_sample_song_builder_program.MusicSampleLearningModel import MusicSampleLearningModel
from ebeat.music_sample_song_builder_program.MusicSampleLoader import MusicSampleLoader
from ebeat.music_sample_song_builder_program.MusicSampleSongBuilder import MusicSampleSongBuilder


class MusicSampleNextSamplePredictor():

    def __init__(self, input_directory: str, output_directory: str, edge_size: int):
        self.output_directory = output_directory
        self.input_directory = input_directory
        self.edge_size = edge_size

    @classmethod
    def use_existing_model(cls, input_directory: str, output_directory: str, edge_size: int, model_path: str):
        cls.model_path = model_path
        return cls(input_directory, output_directory, edge_size)

    @classmethod
    def create_new_model(cls, input_directory: str, output_directory: str, edge_size: int, batch_size: int,
                         latent_dim: int, epochs: int):
        cls.model_path = ''
        cls.edge_size = edge_size
        cls.batch_size = batch_size
        cls.latent_dim = latent_dim
        cls.epochs = epochs
        return cls(input_directory, output_directory, edge_size)

    def create_new_music_file(self):
        loader = MusicSampleLoader(self.input_directory)
        end_samples, start_samples, sample_model = loader.load_samples(self.edge_size)
        learning_model = MusicSampleLearningModel(end_samples, start_samples)
        sample_order = learning_model.predict_sample_order(sample_model, self.model_path)
        song_builder = MusicSampleSongBuilder(self.input_directory, self.output_directory)
        song_builder.save_song(sample_order)
