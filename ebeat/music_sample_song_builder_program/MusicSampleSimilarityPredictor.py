import difflib


class MusicSampleSimilarityPredictor:

    def __init__(self, decoder, sample_model, end_samples, encoder_input_data):
        self.encoder_input_data = encoder_input_data
        self.end_samples = end_samples
        self.decoder = decoder
        self.sample_model = sample_model
        self.already_used_samples = set()
        self.max_length = len(sample_model)
        self.counter = 1

    def predict_next_samples_recursive(self, index):
        input_seq = self.encoder_input_data[index: index + 1]
        decoded_sentence = self.decoder.decode_sequence(input_seq)
        ratio = 0.0
        name = ''
        next_end = []
        for sample in self.sample_model:
            sm = difflib.SequenceMatcher(None, sample.start.tolist(), decoded_sentence)
            new_ratio = sm.ratio()
            if (new_ratio > ratio) and (sample.name not in self.already_used_samples):
                ratio = new_ratio
                name = sample.name
                next_end = sample.end
        self.counter += 1
        if name != '':
            print('predicted sample: ' + name + ' with ratio ' + str(ratio))
            self.already_used_samples.add(name)
        if ratio == 0.0:
            print('No further sample found after ' + str(len(self.already_used_samples)) + ' samples.')
        elif self.counter < self.max_length:
            new_index = self.end_samples.index(next_end)
            self.predict_next_samples_recursive(new_index)