from itertools import islice

import tf

import SamplingConstants


class SampleLoader:

    def load_single_sample(self, is_tf: bool):
        if is_tf:
            return self.__load_samples_tf(SamplingConstants.IN_SINGLE_SAMPLE)
        else:
            return None

    def load_all_samples(self, is_tf: bool):
        if islice:
            return self.__load_samples_tf(SamplingConstants.IN_DIRECTORY)
        else:
            return None

    def __load_samples_tf(self, path: str):
        test_file = tf.io.read_file(path)
        test_audio = tf.audio.decode_wav(contents=test_file)
        return test_audio

    def __load_samples(self, path: str):
        test_file = tf.io.read_file(path)
        test_audio = tf.audio.decode_wav(contents=test_file)
        return test_audio
