import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf


from tensorflow.keras import layers
from tensorflow.keras import models
#from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

test_file = tf.io.read_file('C:/Users/treib/Documents/HDM/2. Semester/Technology Lab/Musik/Test/Tea K Pea - nauticals_1.wav')
test_audio, _ = tf.audio.decode_wav(contents=test_file)
test_audio.shape
