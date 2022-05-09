'''
Program to insert inputs like in_path, out_path
-> Idea: Input model path if a .hdf5 model file should be used - also a epoch input
'''
import sys

from ebeat.music_sample_song_builder_program.MusicSampleNextSamplePredictor import MusicSampleNextSamplePredictor


def run():
    print('Please insert your input directory of samples (like: C:/Users/Admin/Downloads/input/:).')
    input_directory = input()
    print(
        'Please insert your output directory where your song should be saved (like: C:/Users/Admin/Downloads/output/:).')
    output_directory = input()
    print('Build a new model (insert nothing; press enter) or use a existing model (insert: yes).')
    use_existing_model = input()
    if use_existing_model == 'yes':
        print(
            'Please insert your input directory of your .hdf5 model (like: C:/Users/Admin/Downloads/input/model.hdf5:).')
        model_path = input()
        predictor = MusicSampleNextSamplePredictor(input_directory, output_directory, model_path)
        predictor.create_new_music_file()
    else:
        sys.exit()


if __name__ == "__main__":
    run()