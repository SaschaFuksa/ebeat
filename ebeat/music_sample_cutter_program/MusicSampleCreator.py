from ebeat.music_sample_cutter_program.EqualMusicSampleCutter import EqualMusicSampleCutter
from ebeat.music_sample_cutter_program.MusicFileCollector import MusicFileCollector
from ebeat.music_sample_cutter_program.MusicSampleCutter import MusicSampleCutter
from ebeat.music_sample_cutter_program.SampleSaver import SampleSaver
from ebeat.music_sample_cutter_program.StochasticMusicSampleCutter import StochasticMusicSampleCutter

"""
MusicSampleCreator creates samples either on equal or stochastic method.
Samples sizes are based on the cutting method which is chosen, the original file name is used in each
part for traceability. 
"""


class MusicSampleCreator:
    music_sample_cutter: MusicSampleCutter

    def __init__(self, input_directory: str, output_directory: str):
        self.input_directory = input_directory
        self.output_directory = output_directory

    @classmethod
    def equal_cutting_data(cls, input_directory: str, output_directory: str, sample_length: int):
        cls.sample_length = sample_length
        return cls(input_directory, output_directory)

    @classmethod
    def stochastic_cutting_data(cls, input_directory: str, output_directory: str, amount_of_samples: int,
                                min_silence_length: int, silence_threshold: int):
        cls.amount_of_samples = amount_of_samples
        cls.min_silence_length = min_silence_length
        cls.silence_threshold = silence_threshold
        cls.sample_length = 0
        return cls(input_directory, output_directory)

    def create_samples(self):
        file_collector = MusicFileCollector(self.input_directory)
        songs = file_collector.find_songs()
        rate = 44100

        if self.sample_length > 0:
            self.music_sample_cutter = EqualMusicSampleCutter(self.sample_length)
        else:
            self.music_sample_cutter = StochasticMusicSampleCutter(self.amount_of_samples, self.min_silence_length,
                                                                   self.silence_threshold)

        for song in songs:
            song_name = song.split('.')[0]
            postfix = song.split('.')[1]
            samples = self.music_sample_cutter.cut_music_file(self.input_directory + song)
            sample_saver = SampleSaver(self.output_directory, song_name, postfix, rate)

            for sample in samples:
                sample_saver.save_sample(sample)
