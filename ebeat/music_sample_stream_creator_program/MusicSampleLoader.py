from pathlib import Path

import librosa
import numpy as np
from pydub import AudioSegment
from scipy import signal
from sklearn import preprocessing


class MusicSampleLoader:

    def __init__(self):
        """
        Loads samples as dictionaries with file name as key and resampled array as value
        """
        pass

    @staticmethod
    def __normalize_sample(sample):
        array_of_samples = sample.get_array_of_samples()
        np_arr = np.array(array_of_samples)
        np_arr = np_arr.reshape(1, -1)
        return preprocessing.normalize(np_arr)

    @staticmethod
    def load_training_samples_fixed_resample_rate(sample_path: str, resample_rate: 1000):
        """
        Load samples by given path
        :param sample_path: Path to folder with samples in .wav format
        :param resample_rate: Rate as int to reduce sample rate as fixed value for all samples
        :return: dictionary with  file name as key and resampled array as value
        """
        sample_files = librosa.util.find_files(sample_path, ext=['mp3'])
        sample_files = sorted(sample_files, key=lambda x: int(x.split('_')[-1].split(".")[0]))
        loaded_samples = {}
        for file_name in sample_files:
            sample = AudioSegment.from_mp3(file_name)
            mono_samples = sample.split_to_mono()
            normalized_sample_first_canal = MusicSampleLoader.__normalize_sample(mono_samples[0])[0]
            resampled_sample_first_canal = signal.resample(normalized_sample_first_canal, resample_rate)
            normalized_sample_sec_canal = MusicSampleLoader.__normalize_sample(mono_samples[1])[0]
            resampled_sample_sec_canal = signal.resample(normalized_sample_sec_canal, resample_rate)
            file_name = Path(file_name).stem
            loaded_samples[file_name] = [resampled_sample_first_canal, resampled_sample_sec_canal]
        return loaded_samples

    @staticmethod
    def load_training_samples_fixed_resample_rate(sample_path: str, reduction_rate: 35):
        """
        Load samples by given path
        :param sample_path: Path to folder with samples in .wav format
        :param reduction_rate: Rate as int to reduce sample rate like rate = 10, origin = 1000, new rate = origin/rate -> 100
        :return: dictionary with  file name as key and resampled array as value
        """
        sample_files = librosa.util.find_files(sample_path, ext=['mp3'])
        sample_files = sorted(sample_files, key=lambda x: int(x.split('_')[-1].split(".")[0]))
        loaded_samples = {}
        for file_name in sample_files:
            sample = AudioSegment.from_mp3(file_name)
            mono_samples = sample.split_to_mono()
            normalized_sample_first_canal = MusicSampleLoader.__normalize_sample(mono_samples[0])[0]
            resampled_sample_first_canal = signal.resample(normalized_sample_first_canal,
                                                           int(len(normalized_sample_first_canal) / reduction_rate))
            normalized_sample_sec_canal = MusicSampleLoader.__normalize_sample(mono_samples[1])[0]
            resampled_sample_sec_canal = signal.resample(normalized_sample_sec_canal,
                                                         int(len(normalized_sample_sec_canal) / reduction_rate))
            file_name = Path(file_name).stem
            loaded_samples[file_name] = [resampled_sample_first_canal, resampled_sample_sec_canal]
        return loaded_samples
