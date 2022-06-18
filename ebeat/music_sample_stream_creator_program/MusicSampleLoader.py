from pathlib import Path

import librosa
from pydub import AudioSegment
from scipy import signal


class MusicSampleLoader:

    def __init__(self):
        """
        Loads samples as dictionaries with file name as key and resampled array as value
        """
        pass

    @staticmethod
    def load_training_samples(sample_path: str, resample_rate: 15):
        """
        Load samples by given path
        :param sample_path: Path to folder with samples in .wav format
        :param resample_rate: Rate as int to reduce sample rate
        :return: dictionary with  file name as key and resampled array as value
        """
        sample_files = librosa.util.find_files(sample_path, ext=['mp3'])
        sample_files = sorted(sample_files, key=lambda x: int(x.split('_')[-1].split(".")[0]))
        loaded_samples = {}
        for file_path in sample_files:
            loaded_audio_file = AudioSegment.from_mp3(file_path)
            sample_array = loaded_audio_file.get_array_of_samples()
            resampled_part = signal.resample(sample_array, int(len(sample_array) / resample_rate))
            file_name = Path(file_path).stem
            loaded_samples[file_name] = resampled_part
        return loaded_samples
