'''
Program to insert inputs like in_path, out_path
-> Idea: Input model path if a .hdf5 model file should be used - also a epoch input
'''

from ebeat.music_sample_song_builder_program.MusicSampleNextSamplePredictor import MusicSampleNextSamplePredictor


def run():
    print('Please insert your input directory of samples (like: C:/Users/Admin/Downloads/input/:).')
    input_directory = input()
    print(
        'Please insert your output directory where your song should be saved (like: C:/Users/Admin/Downloads/output/:).')
    output_directory = input()
    print('Edge size of samples. If you left 0, default of 50 will be used.):')
    edge_size = int(input())
    if edge_size == 0:
        edge_size += 50
    print('Use callback functions to save model? Insert y for yes:')
    use_callback = input()
    print(
        'Use a existing model (insert: input directory of your .hdf5 model' +
        'like: C:/Users/Admin/Downloads/input/model.hdf5:) or press enter.')
    model_path = input()
    if model_path == '':
        print('Batch size of samples. If you left 0, default of 2 will be used.:')
        batch_size = int(input())
        if batch_size == 0:
            batch_size += 2
        print('Dimension of neurons. If you left 0, default of 1000 will be used.')
        latent_dim = int(input())
        if latent_dim == 0:
            latent_dim += 1000
        print('Epochs for run of training. If you left 0, default of 100 will be used.')
        epochs = int(input())
        if epochs == 0:
            epochs += 100
        predictor = MusicSampleNextSamplePredictor.create_new_model(input_directory, output_directory, edge_size,
                                                                    batch_size, latent_dim, epochs)
        predictor.create_new_music_file()
    else:
        predictor = MusicSampleNextSamplePredictor.use_existing_model(input_directory, output_directory, edge_size,
                                                                      model_path)
        predictor.create_new_music_file()


if __name__ == "__main__":
    run()
