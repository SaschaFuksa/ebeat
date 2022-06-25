from numpy import random

class MusicSampleDataPreparer:

    def __init__(self):
        pass

    @staticmethod
    def get_strange_end_part(samples, current_index, max_len, edge_size):
        """
        Find and add not neighbour samples at the end of reference
        :param samples: samples to find strange part
        :param current_index: Current index to check
        :param max_len: max length of all samples
        :param edge_size: Size of end and start edges
        :return: A sample part as second half which is further away than direct neighbour samples
        """
        x = random.randint(max_len - 1)
        if (x != current_index + 1) and (x != current_index + 2):
            return list(samples[x][:edge_size])
        else:
            return MusicSampleDataPreparer.get_strange_end_part(samples, current_index, max_len, edge_size)

    @staticmethod
    def prepare_valid_data(samples, edge_size):
        """
        Create valid (True) test data
        :param samples: samples to prepare
        :param edge_size: Size of end and start edges
        :return: x and y data
        """
        x = []
        y = []
        for i in range(len(samples) - 1):
            y.append(True)
            first_half = list(samples[i][-edge_size:])
            last_half = list(samples[i + 1][:edge_size])
            x.append(first_half + last_half)
            if i < len(samples) - 2:
                y.append(True)
                sec_last_half = list(samples[i + 2][:edge_size])
                x.append(first_half + sec_last_half)
        return x, y

    @staticmethod
    def prepare_invalid_data(samples, edge_size):
        """
        Create invalid (False) test data
        :param samples: samples to prepare
        :param edge_size: Size of end and start edges
        :return: x and y data
        """
        x = []
        y = []
        for i in range(len(samples) - 1):
            first_half = list(samples[i][-edge_size:])
            last_half = MusicSampleDataPreparer.get_strange_end_part(samples, i, len(samples), edge_size)
            x.append(first_half + last_half)
            y.append(False)
            sec_last_half = MusicSampleDataPreparer.get_strange_end_part(samples, i, len(samples), edge_size)
            x.append(first_half + sec_last_half)
            y.append(False)
        return x, y

    @staticmethod
    def prepare_data(samples, samples_sec_canal, edge_size):
        """
        Create x_train, y_train, x_val and y_val data
        :param samples: samples of first canal
        :param samples_sec_canal: samples of second canal
        :param edge_size: Size of end and start edges
        :return: x_train, y_train, x_val and y_val data
        """
        x_train, y_train = MusicSampleDataPreparer.prepare_valid_data(samples, edge_size)
        x_val, y_val = MusicSampleDataPreparer.prepare_valid_data(samples_sec_canal, edge_size)

        x_train_false, y_train_false = MusicSampleDataPreparer.prepare_invalid_data(samples, edge_size)
        x_val_false, y_val_false = MusicSampleDataPreparer.prepare_invalid_data(samples_sec_canal, edge_size)

        x_train.extend(x_train_false)
        y_train.extend(y_train_false)
        x_val.extend(x_val_false)
        y_val.extend(y_val_false)
        return x_train, y_train, x_val, y_val
