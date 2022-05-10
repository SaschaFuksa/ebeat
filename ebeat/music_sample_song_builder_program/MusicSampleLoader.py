'''
Program to load all samples of in_path and prepare/compiles (normalize) data e.g. start/end 
'''
import os

from pydub import AudioSegment

from ebeat.music_sample_song_builder_program.MusicSampleModel import MusicSampleModel


class MusicSampleLoader:

    def __init__(self, input_directory: str):
        self.input_directory = input_directory

    def load_samples(self, edge_size: int):
        start_samples = []
        end_samples = []
        files = os.listdir(self.input_directory)
        files = sorted(files, key=lambda x: int(x.split('_')[-1].split(".")[0]))
        sample_model = []
        for file_name in files:
            if '.wav' in file_name:
                complete_path = self.input_directory + file_name
                sample = AudioSegment.from_wav(complete_path)
                numeric_sample_array = sample.get_array_of_samples()
                start_samples.append(numeric_sample_array[:edge_size])
                end_samples.append(numeric_sample_array[-edge_size:])
                model = MusicSampleModel(name=file_name, start=numeric_sample_array[:edge_size],
                                         end=numeric_sample_array[-edge_size:])
                sample_model.append(model)
        end_samples = end_samples[:-1]
        start_samples = start_samples[1:]

        return end_samples, start_samples, sample_model
