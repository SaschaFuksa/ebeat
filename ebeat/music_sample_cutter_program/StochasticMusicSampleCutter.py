from pydub import AudioSegment
from pydub.silence import split_on_silence

from ebeat.music_sample_cutter_program.MusicSampleCutter import MusicSampleCutter


class StochasticMusicSampleCutter(MusicSampleCutter):

    def __init__(self, amount_of_samples: int, min_silence_len: int, silence_thresh: int):
        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len
        self.amount_of_samples = amount_of_samples

    def cut_music_file(self, music_file_path: str) -> []:
        song = AudioSegment.from_wav(music_file_path)
        samples = split_on_silence(song, self.min_silence_len, self.silence_thresh)
        return samples
