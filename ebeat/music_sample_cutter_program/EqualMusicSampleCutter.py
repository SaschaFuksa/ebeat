import scipy.io.wavfile
import librosa
import pydub
import numpy as np

from ebeat.music_sample_cutter_program.MusicSampleCutter import MusicSampleCutter


class EqualMusicSampleCutter(MusicSampleCutter):
    """EqualMusicSampleCutter cuts a music file to equal-sized sample parts"""

    def __init__(self, sample_length: int):
        self.sample_length = sample_length

    def cut_music_file(self, music_file_path: str) -> []:
        samples = []
        rate = 1250
        #rate, audio_data = scipy.io.wavfile.read(music_file_path)
        #audio_data, rate = librosa.load(music_file_path)
        audio_data = pydub.AudioSegment.from_mp3(music_file_path)
        #np_array = np.array(audio_data.get_array_of_samples())
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
