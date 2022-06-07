import librosa
import numpy as np
import tensorflow as tf

#def createDataset():
from pydub import AudioSegment

pathAudio = 'C:/Users/Admin/Downloads/Technology Lab/sampling/in-1sec/'
files = librosa.util.find_files(pathAudio, ext=['wav'])
files = np.asarray(files)
A = []
#print(files)
for file_name in files:
    #complete_path = pathAudio + file_name
    sample = AudioSegment.from_wav(file_name)
    numeric_sample_array = sample.get_array_of_samples()
    #test_file = tf.io.read_file(y)
    #test_audio, _ = tf.audio.decode_wav(contents=test_file)
    #numpyvari = test_audio.numpy()
    A.append(numeric_sample_array)
    #print(test_audio.shape)
A = np.array(A)
print(len(A))
'''print(A.shape)
print(A.max())
print(A.min())


def normalize_data_2(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

normalized_data_2 = normalize_data_2(A)
print(normalized_data_2.max())
print(normalized_data_2.min())

def normalize_data(data, x_max, x_min, d1, d2):
    return ((data-x_min)*(d2-d1)/(x_max-x_min))+d1

normalized_data = normalize_data(A, A.max(), A.min(), 0, 1)
print(normalized_data.max())
print(normalized_data.min())'''

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

scaled_x = min_max_scaler.fit_transform(A)
#print(scaled_x.max())
#print(scaled_x.min())
#print(scaled_x)
print(scaled_x.shape)

reshaped_x = scaled_x.reshape(222, 88200, 1)
print(reshaped_x.shape)

from matplotlib import pyplot as plt

plt.figure(1)
plt.plot(scaled_x[-1])
plt.show()

train_x = reshaped_x[:190]
print((train_x.shape))
test_x = reshaped_x[190:]
print((test_x.shape))

