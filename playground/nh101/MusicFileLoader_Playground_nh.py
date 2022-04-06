import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import tensorflow_io as tfio
from IPython.display import Audio

path = "/Users/nicolahaller/Documents/WI Master/Semester 2/Technology Lab/Tea K Pea - nauticals.mp3"

audio = tfio.audio.AudioIOTensor(path)
print('audio tensor:')
print(audio)
print(type(audio))

audio_slice = audio[100000:110000]
print('audio slice:')
print(audio_slice)
Audio(audio_slice.numpy(), rate=audio.rate.numpy())

tensor = tf.cast(audio_slice, tf.float32) / 32768.0
tensor_numpy = tensor.numpy()
print('audio numpy:')
print(tensor_numpy)
print(type(tensor_numpy))
#tensor_list = list(tensor_numpy)
#print(tensor_list)
#print(type(tensor_list))

plt.figure()
plt.plot(tensor_numpy)
plt.show()

first_dim = numpy.delete(tensor_numpy, 1)
print('first dim:')
print(first_dim)
print(type(first_dim))

second_dim = numpy.delete(tensor_numpy, 0)
print('second dim:')
print(second_dim)

plt.plot(first_dim)
#plt.show()

plt.plot(second_dim)
#plt.show()