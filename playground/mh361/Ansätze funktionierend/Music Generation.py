import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


#read the data my_song.abc
with open('my_song.abc', 'r') as f:
    text = f.read()

#unique symbols
vocab = set(text)
#char encoding
char_to_index = {char_ :ind for ind, char_  in enumerate (vocab)} ind_to_char = np.array(vocab)
text_as_int = np.array([char_to_index[c] for c in text])
# 'X:1\nT:dfkjds ' ----- > [49 22 13  0 45 22 26 67 60 79 56 69 59]

seq_length = 100
step = 10
sequences = np.array([text_as_int[i:i+seq_length+1] for i in range(0, len(text_as_int)-seq_length-1,step)])
input_text = np.array([seq[:-1] for seq in sequences])
target_text = np.array([seq[1:] for seq in sequences])