# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import models
# from IPython import display

# Load the chopped sample files and create a Tensor out of it
Sample_File = tf.io.read_file('C:/Users/hennm/Dropbox/Studium/Master/Semester 2/Tec Lab/Music/WAV/Tea K Pea - '
                              'nauticals.wav')
Sample_File, _ = tf.audio.decode_wav(contents=Sample_File)
print(Sample_File.shape)