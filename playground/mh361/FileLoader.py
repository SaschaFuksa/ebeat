import tensorflow
import pathlib
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import librosa

music_dir = 'C:\\Users\\hennm\\Dropbox\\Studium\\Master\\Semester 2\\Tec Lab\\Music\\'

wav_filenames = glob.glob(str(music_dir + '*.wav*'))
wav_file = wav_filenames[0]
mp3_filenames = glob.glob(str(music_dir + '*.mp3*'))
mp3_file = mp3_filenames[0]

# File Pre-Fix name
file_prefix = os.path.basename(wav_file).split('.')[0]

# WAV File
song = AudioSegment.from_wav(wav_file)

# MP3 File
#song = AudioSegment.from_mp3(mp3_file)

# Other File Formats
"""
mp4_version = AudioSegment.from_file("never_gonna_give_you_up.mp4", "mp4")
wma_version = AudioSegment.from_file("never_gonna_give_you_up.wma", "wma")
aac_version = AudioSegment.from_file("never_gonna_give_you_up.aiff", "aac")

# first attempt

dname = 'C:\\Users\\hennm\\Dropbox\\Studium\\Master\\Semester 2\\Tec Lab\\Music\\MP3\\'
filenames = glob.glob(str(dname+'*.mp3*'))
sample_file = filenames[1]

audio = tfio.audio.AudioIOTensor(sample_file)
print(audio)
x , sr = librosa.load(audio)
#print(type(x), type(sr))#<class 'numpy.ndarray'> <class 'int'>print(x.shape, sr)#(94316,) 2205


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