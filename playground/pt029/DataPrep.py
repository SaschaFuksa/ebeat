import librosa
import numpy as np
import tensorflow as tf

#def createDataset():
pathAudio = "C:/Users/treib/Documents/HDM/2. Semester/Technology Lab/Musik/Test/"
files = librosa.util.find_files(pathAudio, ext=['wav'])
files = np.asarray(files)
A = []

for y in files:
    test_file = tf.io.read_file(y)
    test_audio, _ = tf.audio.decode_wav(contents=test_file)
    numpyvari = test_audio.numpy()
    print(numpyvari.min())
    print(test_audio.shape)