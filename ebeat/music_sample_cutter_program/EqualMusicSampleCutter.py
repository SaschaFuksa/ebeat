import scipy.io.wavfile

from ebeat.music_sample_cutter_program.MusicSampleCutter import MusicSampleCutter

"""
EqualMusicSampleCutter cuts a music file to equal-sized sample parts
"""


class EqualMusicSampleCutter(MusicSampleCutter):

    def __init__(self, sample_length: int):
        self.sample_length = sample_length

    def cut_music_file(self, music_file_path: str) -> []:
        samples = []
        rate, audio_data = scipy.io.wavfile.read(music_file_path)
        time = len(audio_data) / rate
        amount_of_samples = time / self.sample_length
        sample_size = int(self.sample_length * rate)
        if amount_of_samples >= 1:
            i = 0
            while i < len(audio_data):
                sample = audio_data[i:i + sample_size]
                samples.append(sample)
                i += sample_size
        return samples
