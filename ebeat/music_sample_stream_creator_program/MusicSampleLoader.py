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

        song_lounge_it = []
        song_tuesday_night = []
        song_kingtop = []
        for file in sample_files:
            if 'Maarten Schellekens - Lounge It' in file:
                song_lounge_it.append(file)
            if 'Maarten Schellekens - Tuesday Night Radio Edit' in file:
                song_tuesday_night.append(file)
            if 'Tea K Pea - kingtop' in file:
                song_kingtop.append(file)
        song_lounge_it = sorted(song_lounge_it, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        song_tuesday_night = sorted(song_tuesday_night, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        song_kingtop = sorted(song_kingtop, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        full_songs = []
        full_songs.extend(song_lounge_it)
        full_songs.extend(song_tuesday_night)
        full_songs.extend(song_kingtop)

        samples = []
        samples_sec_canal = []
        for file_name in full_songs:
            sample = AudioSegment.from_mp3(file_name)
            mono_samples = sample.split_to_mono()
            normalized_sample = MusicSampleLoader.__normalize_sample(mono_samples[0])[0]
            resampled_sample = signal.resample(normalized_sample, resample_rate)
            samples.append(resampled_sample)
            normalized_sample_sec_canal = MusicSampleLoader.__normalize_sample(mono_samples[1])[0]
            resampled_sample_sec_canal = signal.resample(normalized_sample_sec_canal, resample_rate)
            samples_sec_canal.append(resampled_sample_sec_canal)

        return samples, samples_sec_canal

    @staticmethod
    def load_training_samples_fixed_resample_rate(sample_path: str, reduction_rate: 35):
        """
        Load samples by given path
        :param sample_path: Path to folder with samples in .wav format
        :param reduction_rate: Rate as int to reduce sample rate like rate = 10, origin = 1000, new rate = origin/rate -> 100
        :return: dictionary with  file name as key and resampled array as value
        """
        sample_files = librosa.util.find_files(sample_path, ext=['mp3'])

        song_lounge_it = []
        song_tuesday_night = []
        song_kingtop = []
        for file in sample_files:
            if 'Maarten Schellekens - Lounge It' in file:
                song_lounge_it.append(file)
            if 'Maarten Schellekens - Tuesday Night Radio Edit' in file:
                song_tuesday_night.append(file)
            if 'Tea K Pea - kingtop' in file:
                song_kingtop.append(file)
        song_lounge_it = sorted(song_lounge_it, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        song_tuesday_night = sorted(song_tuesday_night, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        song_kingtop = sorted(song_kingtop, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        full_songs = []
        full_songs.extend(song_lounge_it)
        full_songs.extend(song_tuesday_night)
        full_songs.extend(song_kingtop)

        samples = []
        samples_sec_canal = []
        for file_name in full_songs:
            sample = AudioSegment.from_mp3(file_name)
            mono_samples = sample.split_to_mono()
            normalized_sample = MusicSampleLoader.__normalize_sample(mono_samples[0])[0]
            resampled_sample = signal.resample(normalized_sample, int(len(normalized_sample) / reduction_rate))
            samples.append(resampled_sample)
            normalized_sample_sec_canal = MusicSampleLoader.__normalize_sample(mono_samples[1])[0]
            resampled_sample_sec_canal = signal.resample(normalized_sample_sec_canal, int(len(normalized_sample) / reduction_rate))
            samples_sec_canal.append(resampled_sample_sec_canal)
        return samples, samples_sec_canal


    @staticmethod
    def load_sample_pool_fixed_resample_rate(sample_path: str, reduction_rate: 35, edge_size):
        """
        Load all samples of sample pool as dic
        :param sample_path: Path to sample pool
        :param reduction_rate: rate to reduce
        :param edge_size: end and start edge size
        :return: samples as dic, key = title, value = array
        """
        sample_files = librosa.util.find_files(sample_path, ext=['mp3'])
        loaded_samples = {}
        for file_path in sample_files:
            loaded_audio_file = AudioSegment.from_mp3(file_path)
            mono_samples = loaded_audio_file.split_to_mono()
            normalized_sample = MusicSampleLoader.__normalize_sample(mono_samples[0])[0]
            resampled_part = signal.resample(normalized_sample, int(len(normalized_sample) / reduction_rate))
            if len(resampled_part) >= edge_size:
                file_name = Path(file_path).stem
                loaded_samples[file_name] = resampled_part
        return loaded_samples