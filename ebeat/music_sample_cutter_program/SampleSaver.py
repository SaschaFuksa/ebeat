import numpy
from scipy.io.wavfile import write


class SampleSaver:

    def __init__(self, directory: str, track_name: str, audio_type: str, rate: int):
        self.directory = directory
        self.track_name = track_name
        self.sample_counter = 1
        self.audio_type = audio_type
        self.rate = rate

    def save_sample(self, audio_data):
        target_name = self.directory + self.track_name + "_" + str(self.sample_counter) + "." + self.audio_type
        write(target_name, self.rate, audio_data.astype(numpy.int16))
        self.sample_counter += 1
