import numpy 
 
from ebeat.music_sample_song_builder_program.MusicSampleDecoderModel import MusicSampleDecoderModel 
 
 
class MusicSampleDecoder: 
 
    def __init__(self, encoder_model, decoder_model, decoder_model_builder: MusicSampleDecoderModel, 
                 target_token_index): 
        self.decoder_model = decoder_model 
        self.encoder_model = encoder_model 
        self.decoder_model_builder = decoder_model_builder 
        self.reverse_target_char_index = dict( 
            (i, float32) for float32, i in target_token_index.items()) 
 
    def decode_sequence(self, input_seq): 
        states_value = self.encoder_model.predict(input_seq) 
        target_seq = numpy.zeros((1, 1, self.decoder_model_builder.num_decoder_tokens)) 
        stop_condition = False 
        decoded_sentence = [] 
        while not stop_condition: 
            output_tokens, h, c = self.decoder_model.predict( 
                [target_seq] + states_value) 
            sampled_token_index = numpy.argmax(output_tokens[0, -1, :]) 
            sampled_char = self.reverse_target_char_index[sampled_token_index] 
            decoded_sentence.append(sampled_char) 
            if len(decoded_sentence) >= self.decoder_model_builder.max_decoder_seq_length: 
                stop_condition = True 
            target_seq = numpy.zeros((1, 1, self.decoder_model_builder.num_decoder_tokens)) 
            target_seq[0, 0, sampled_token_index] = 1. 
            states_value = [h, c] 
 
        return decoded_sentence 

 