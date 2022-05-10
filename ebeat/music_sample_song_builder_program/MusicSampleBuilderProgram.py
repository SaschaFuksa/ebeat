'''
Program to insert inputs like in_path, out_path
-> Idea: Input model path if a .hdf5 model file should be used - also a epoch input
'''
from ebeat.music_sample_song_builder_program.MusicSampleConfiguration import MusicSampleConfiguration
from ebeat.music_sample_song_builder_program.MusicSampleNextSamplePredictor import MusicSampleNextSamplePredictor


def run():
    print('Please insert your input directory of samples (like: C:/Users/Admin/Downloads/input/:).')
    MusicSampleConfiguration.input_directory = input()
    if MusicSampleConfiguration.input_directory == 'sf':
        MusicSampleConfiguration.input_directory = 'C:/Users/sasch/OneDrive/Dokumente/Studium/Technology Lab/Techno Titel/samples_in/'
        MusicSampleConfiguration.output_directory = 'C:/Users/sasch/OneDrive/Dokumente/Studium/Technology Lab/Techno Titel/music_out/'
        MusicSampleConfiguration.model_path = ''
    else:
        print(
            'Please insert your output directory where your song should be saved ' +
            '(like: C:/Users/Admin/Downloads/output/:).')
        MusicSampleConfiguration.output_directory = input()
        print('Edge size of samples. If you left 0, default of 50 will be used.):')
        edge_size = int(input())
        if edge_size > 0:
            MusicSampleConfiguration.edge_size += 50
        print('Use callback functions to save model? Insert y for yes:')
        use_callback = input()
        if use_callback == 'yes':
            MusicSampleConfiguration.use_callback = True
        print(
            'Use a existing model (insert: input directory of your .hdf5 model' +
            'like: C:/Users/Admin/Downloads/input/model.hdf5:) or press enter.')
        MusicSampleConfiguration.model_path = input()
        if MusicSampleConfiguration.model_path == '':
            print('Batch size of samples. If you left 0, default of 2 will be used.:')
            batch_size = int(input())
            if batch_size > 0:
                MusicSampleConfiguration.batch_size = batch_size
            print('Dimension of neurons. If you left 0, default of 1000 will be used.')
            latent_dim = int(input())
            if latent_dim > 0:
                MusicSampleConfiguration.latent_dim = latent_dim
            print('Epochs for run of training. If you left 0, default of 100 will be used.')
            epochs = int(input())
            if epochs > 0:
                MusicSampleConfiguration.epochs = epochs
    MusicSampleNextSamplePredictor.create_new_music_file()


if __name__ == "__main__":
    run()
