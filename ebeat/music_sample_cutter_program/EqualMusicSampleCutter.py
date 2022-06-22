import scipy.io.wavfile
import librosa
import pydub
import numpy as np
from pydub import AudioSegment

from ebeat.music_sample_cutter_program.MusicSampleCutter import MusicSampleCutter


class EqualMusicSampleCutter(MusicSampleCutter):
    """EqualMusicSampleCutter cuts a music file to equal-sized sample parts"""

    def __init__(self, sample_length: int):
        self.sample_length = sample_length

    def cut_music_file(self, music_file_path: str) -> []:
        samples = []
        song = AudioSegment.from_mp3(music_file_path)
        time = song.duration_seconds
        amount_of_samples = time / self.sample_length
        sample_size = int(self.sample_length * 1000)
        if amount_of_samples >= 1:
            i = 0
            while i < len(song):
                sample = song[i:i + sample_size]
                samples.append(sample)
                i += sample_size
        return samples
