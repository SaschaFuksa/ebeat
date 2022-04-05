import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import tensorflow_io as tfio
from IPython.display import Audio

path = "C:\\Users\\Admin\\OneDrive\\Dokumente\\Studium\\Technology Lab\\Techno Titel\\Tea K Pea - nauticals.mp3"

audio = tfio.audio.AudioIOTensor(path)
print('audio tensor:')
print(audio)
print(type(audio))

audio_slice = audio[10000:50000]
print('audio slice:')
print(audio_slice)
print(type(audio_slice))
#Audio(audio_slice.numpy(), rate=audio.rate.numpy())

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
#plt.show()

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

from scipy.io.wavfile import write

#write('C:\\Users\\Admin\\OneDrive\\Dokumente\\Studium\\Technology Lab\\Techno Titel\\Test2.wav', 44100, tensor_numpy)
write('C:\\Users\\Admin\\OneDrive\\Dokumente\\Studium\\Technology Lab\\Techno Titel\\Test2.wav', 44100, tensor_numpy)