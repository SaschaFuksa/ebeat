import numpy
from pydub import AudioSegment
from scipy.io.wavfile import write


class SampleSaver:
    """SampleSaver is used to save the samples which are either created by equal or stochastic approach."""

    # Init method which is called as soon as an object of a class is instantiated.
    def __init__(self, directory: str, track_name: str, audio_type: str, rate: int):
        self.directory = directory
        self.track_name = track_name
        self.sample_counter = 1
        self.audio_type = audio_type
        self.rate = rate

    # Function to save the samples after they are cut either by equal or stochastic approach
    def save_sample(self, audio_data):
        target_name = self.directory + self.track_name + "_" + str(self.sample_counter) + "." + self.audio_type
        if isinstance(audio_data, AudioSegment):
            audio_data.export(target_name, format=self.audio_type)
        else:
            write(target_name, self.rate, audio_data.astype(numpy.int16))
        self.sample_counter += 1
