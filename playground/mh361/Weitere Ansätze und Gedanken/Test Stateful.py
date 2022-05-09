import numpy as np
import os
import Paths
import librosa
import tensorflow as tf
from keras import models, layers
import time


def define_model(len_ts,
                 hidden_neurons = 10,
                 nfeature=1,
                 batch_size=None,
                 stateful=False):
    in_out_neurons = 1

    inp = layers.Input(batch_shape= (batch_size, len_ts, nfeature),
                       name="input")

    rnn = layers.LSTM(hidden_neurons,
                    return_sequences=True,
                    stateful=stateful,
                    name="RNN")(inp)

    dens = layers.TimeDistributed(layers.Dense(in_out_neurons,name="dense"))(rnn)
    model = models.Model(inputs=[inp],outputs=[dens])

    model.compile(loss="mean_squared_error",
                  sample_weight_mode="temporal",
                  optimizer="rmsprop")
    return(model,(inp,rnn,dens))




directory = os.listdir(Paths.IN_DIRECTORY)
array = []

sequence_length = []
for sample_name in directory:
    if 'wav' in sample_name:
        file_path = os.path.join(Paths.IN_DIRECTORY, sample_name)
        song, sr = librosa.load(file_path)
        sequence_length.append(len(song))
        #print(len(song))
        #song = AudioSegment.from_wav(file_path)
        #samples = song.get_array_of_samples()
        array.append(song)

# Removing the last element using slicing
lastElementIndex = len(array)-1
in_array = array[:lastElementIndex]
out_array = array[1:]

# Get overall number of Input values from samples
#Number_of_values = sum(x for x in sequence_length)


# Create the Input and Output to tensors
input_tensor = tf.convert_to_tensor(in_array)
output_tensor = tf.convert_to_tensor(out_array)

batch_size = len(input_tensor)
timestep = sequence_length[0]

input_data = tf.reshape(input_tensor, shape=(batch_size, timestep, 1))
output_data = tf.reshape(output_tensor, shape=(batch_size, timestep, 1))

prop_train = 0.8
ntrain = int(input_data.shape[0]*prop_train)

D=100

w = np.zeros(output_data.shape[:2])
w[:,D:] = 1
w_train = w

X_train, X_val = input_data[:ntrain], input_data[ntrain:]
y_train, y_val = output_data[:ntrain], output_data[ntrain:]
w_train, w_val = w[:ntrain], w[ntrain:]

hunits = 64

model_stateful, _ = define_model(
    hidden_neurons = hunits,
    batch_size=400,
    stateful=True,
    len_ts = 500)

#model_stateful.summary()

hunits = 64
model_stateless, _ = define_model(
                    hidden_neurons = hunits,
                    len_ts = X_train.shape[1])
model_stateless.summary()

start = time.time()
history = model_stateless.fit(X_train,y_train,
                             batch_size=400,
                             epochs=20,
                             verbose=1,
                              sample_weight=w_train,
                             validation_data=(X_val,y_val,w_val))
end = time.time()
print("Time Took :{:3.2f} min".format( (end-start)/60 ))

'''

smodel = StatefulModel(model=model_stateful, print_val_every=500)

start = time.time()
smodel.fit(X_train, y_train, w_train,
                              X_val, y_val, w_val,
                              Nepoch=100)

end = time.time()
print("Time Took {:3.2f} min".format((end - start) / 60))
'''