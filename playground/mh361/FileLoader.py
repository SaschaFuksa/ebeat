import tensorflow
import pathlib
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import librosa

dname = 'C:\\Users\\hennm\\Dropbox\\Studium\\Master\\Semester 2\\Tec Lab\\Music\\MP3\\'
filenames = glob.glob(str(dname+'*.mp3*'))
sample_file = filenames[1]

audio = tfio.audio.AudioIOTensor(sample_file)
print(audio)
x , sr = librosa.load(audio)
#print(type(x), type(sr))#<class 'numpy.ndarray'> <class 'int'>print(x.shape, sr)#(94316,) 2205

"""
audio = tfio.audio.AudioIOTensor(sample_file)
print(audio)

audio_slice = audio[100:]
print(audio_slice)

tensor = tf.cast(audio_slice, tf.float32) / 32768.0

plt.figure()
plt.plot(tensor.numpy())
plt.show()
print(tensor.numpy())
"""