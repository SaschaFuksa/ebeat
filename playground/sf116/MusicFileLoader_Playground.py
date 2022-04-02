# Ziel: Musikdatei laden, decodieren -> Tensorflow bietet hierfür Methoden an
# Decodierte Musikdatei: Was habe ich für eine Datenstruktur in der Hand? Wie erhalte ich eine nummerische Grundlage für das neuronale Netzwerk?
# Visualisierung: Was kann man aus den decodierten Daten lesen? Ziel: Mustererkennung für stochastische Samples
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

path = "C:\\Users\\Admin\\OneDrive\\Dokumente\\Studium\\Technology Lab\\Lounge Titel\\Dee Yan-Key - minor melancholy.mp3"

audio = tfio.audio.AudioIOTensor(path)
print(audio)

audio_slice = audio[100:]
print(audio_slice)

tensor = tf.cast(audio_slice, tf.float32) / 32768.0

plt.figure()
plt.plot(tensor.numpy())
plt.show()
print(tensor.numpy())
