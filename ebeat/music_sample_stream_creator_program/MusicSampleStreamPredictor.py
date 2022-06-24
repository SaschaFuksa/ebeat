import numpy as np

from ebeat.music_sample_stream_creator_program.MusicSampleConfiguration import MusicSampleConfiguration
from ebeat.music_sample_stream_creator_program.MusicSampleLoader import MusicSampleLoader


class MusicSampleStreamPredictor:

    def __init__(self):
        pass

    @staticmethod
    def predict_classification_stream(reference_part, model):
        """
        Predicts all samples for stream
        :param reference_part: Sample (samples) reference to start with
        :param model: Trained model
        :return: List of song names to build music file
        """
        music_pool = MusicSampleLoader.load_training_samples_fixed_resample_rate(
            MusicSampleConfiguration.sample_pool_path, 35)
        i = 0
        current_class = 0
        reference_sample = reference_part
        max = MusicSampleConfiguration.stream_length
        selected_samples = ['Tea K Pea - nauticals_1']
        while i < max:
            best_sample_name = None
            best_score = 0.00000
            best_sample = []
            for key, value in music_pool.items():
                if key not in selected_samples:
                    ref_and_sample = np.append(reference_sample, value[0])
                    prediction = model.predict(ref_and_sample.reshape(1, 2000))
                    score = prediction[0][current_class]
                    if score > best_score:
                        best_score = score
                        best_sample_name = key
                        best_sample = value[0]
                    if score > 0.95:
                        break
            print('Predicted ' + best_sample_name + ' at class ' + str(
                current_class) + ' as best sample with score ' + str(
                best_score))
            reference_sample = best_sample
            selected_samples.append(best_sample_name)
            i += 1
            current_class += 1
        return selected_samples
