'''
Build song on given suggestion of lstm model by predicted sample names
'''
from pydub import AudioSegment

from ebeat.music_sample_song_builder_program.MusicSampleConfiguration import MusicSampleConfiguration


class MusicSampleSongBuilder:
    def __init__(self):
        pass

    @staticmethod
    def save_song(sample_order):
        new_song = AudioSegment.empty()
        for sample in sample_order:
            complete_path = MusicSampleConfiguration.input_directory + sample
            sample = AudioSegment.from_wav(complete_path)
            new_song += sample
        new_song.export(MusicSampleConfiguration.output_directory + 'karacho.wav', format='wav')
