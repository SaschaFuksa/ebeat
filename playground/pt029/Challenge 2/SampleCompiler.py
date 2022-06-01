import tf as tf
import librosa
import matplotlib.pyplot as plt
import pandas as pd


class SampleCompiler:
    """Class to compile single sample to numeric format."""

    y, sr = librosa.load(
        'C:/Users/treib/Documents/HDM/2. Semester/Technology Lab/Musik/Test/Tea K Pea - nauticals_1.wav')

    df = pd.DataFrame(y, columns=['Amplitude'])

    df.index = [(1 / sr) * i for i in range(len(df.index))]

    # print(df.head())
    # print(df.tail())
    print(df)

    # https://stackoverflow.com/questions/62261039/how-can-i-get-a-dataframe-of-frequency-and-time-from-a-wav-file-in-python

    def compilesample(self):
        data, sampling_rate = librosa.load('C:/Users/treib/Documents/HDM/2. Semester/Technology Lab/Musik/Test')
        # for use in tensorflow
        data_tensor = tf.convert_to_tensor(data)
        return data_tensor

# plt.plot(data_tensor)

# def compilesample
#    tf.audio.encode_wav(
#      audio, sample_rate, name=None)
