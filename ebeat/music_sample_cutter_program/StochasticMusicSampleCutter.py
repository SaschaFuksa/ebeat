from pydub import AudioSegment
from pydub.silence import split_on_silence

from ebeat.music_sample_cutter_program.MusicSampleCutter import MusicSampleCutter

"""
StochasticMusicSampleCutter cuts a music file to silence based sample parts.
The silence parameters as silence_duration can be adjusted in milliseconds.
The silence threshold is used to set a value on which decibel the silence detection should work.
"""


class StochasticMusicSampleCutter(MusicSampleCutter):

    # Init method which is called as soon as an object of a class is instantiated.
    def __init__(self, amount_of_samples: int, min_silence_length: int, silence_threshold: int):
        self.silence_threshold = silence_threshold
        self.min_silence_length = min_silence_length
        self.amount_of_samples = amount_of_samples

    # Function to cut the music files into pieces based on silence_detection
    def cut_music_file(self, music_file_path: str) -> []:
        song = AudioSegment.from_wav(music_file_path)
        samples = split_on_silence(song, self.min_silence_length, self.silence_threshold)
        return samples
