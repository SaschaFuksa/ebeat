from pydub import AudioSegment
from pydub.silence import split_on_silence

from ebeat.music_sample_cutter_program.MusicSampleCutter import MusicSampleCutter


class StochasticMusicSampleCutter(MusicSampleCutter):

    def __init__(self, amount_of_samples: int, min_silence_length: int, silence_threshold: int):
        self.silence_threshold = silence_threshold
        self.min_silence_length = min_silence_length
        self.amount_of_samples = amount_of_samples

    def cut_music_file(self, music_file_path: str) -> []:
        song = AudioSegment.from_wav(music_file_path)
        samples = split_on_silence(song, self.min_silence_length, self.silence_threshold)
        return samples
