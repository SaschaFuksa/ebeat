import numpy as np

from ebeat.music_sample_stream_creator_program.MusicSampleConfiguration import MusicSampleConfiguration
from ebeat.music_sample_stream_creator_program.MusicSampleLoader import MusicSampleLoader


class MusicSampleStreamPredictor:

    def __init__(self):
        pass

    @staticmethod
    def predict_classification_stream(reference_part, model, edge_size):
        """
        Predicts all samples for stream
        :param reference_part: Sample (samples) reference to start with
        :param model: Trained model
        :return: List of song names to build music file
        """
        music_pool = MusicSampleLoader.load_sample_pool_fixed_resample_rate(
            MusicSampleConfiguration.sample_pool_path, 35, edge_size)
        i = 0
        max = 5
        selected_samples = ['Maarten Schellekens - Lounge It_1']
        reference_sample = reference_part
        while i < max:
            best_sample_name = None
            best_score = 0.00000
            best_sample = []
            for key, value in music_pool.items():
                if key not in selected_samples:
                    ref_and_sample = np.append(reference_sample[-edge_size:], value[:edge_size])
                    prediction = model.predict(ref_and_sample.reshape(1, 932))
                    score = prediction[0][0]
                    if score > best_score:
                        best_score = score
                        best_sample_name = key
                        best_sample = value
                    if score > 0.99:
                        break
            print('Predicted ' + best_sample_name + ' at position ' + str(i) + ' as best sample with score ' + str(
                best_score))
            reference_sample = best_sample
            selected_samples.append(best_sample_name)
            i += 1
        return selected_samples
