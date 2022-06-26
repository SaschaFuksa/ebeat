from keras.layers import Conv1D, LSTM, Dense, Flatten, MaxPool1D
from keras.models import Sequential


class MusicSampleModel:

    def __init__(self):
        pass

    @staticmethod
    def create_model(edge_size):
        """
        Creates and complie model
        :param edge_size: Size of end and start edges
        :return: Created model
        """

        model = Sequential()
        model.add(Conv1D(filters=1024, kernel_size=2, input_shape=(2 * edge_size, 1)))
        model.add(MaxPool1D(pool_size=2, strides=2))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dense(256))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        return model
