from ebeat.music_sample_stream_creator_program.MusicSampleConfiguration import MusicSampleConfiguration
from ebeat.music_sample_stream_creator_program.MusicSampleStreamCreator import MusicSampleStreamCreator


def run():
    print('Please insert your input directory of train samples (like: C:/Users/Admin/Downloads/input/:).')
    MusicSampleConfiguration.train_sample_path = input()
    if MusicSampleConfiguration.train_sample_path == 'sf':
        MusicSampleConfiguration.train_sample_path = 'C:/Users/Admin/OneDrive/Dokumente/Studium/Technology Lab/Technology Lab Team 4/Techno Titel/train/samples-stochastic/'
        MusicSampleConfiguration.sample_pool_path = 'C:/Users/Admin/OneDrive/Dokumente/Studium/Technology Lab/Technology Lab Team 4/Techno Titel/music-pool/samples/all_samples/'
        MusicSampleConfiguration.output_directory = 'C:/Users/Admin/OneDrive/Dokumente/Studium/Technology Lab/Technology Lab Team 4/Techno Titel/music_out/'
        MusicSampleConfiguration.model_path = 'C:/Users/sasch/Downloads/model/weights-improvement-215-0.0084-bigger.hdf5'
        MusicSampleConfiguration.use_model = False
        MusicSampleConfiguration.edge_size = 70
        MusicSampleConfiguration.batch_size = 5
        epochs = 100
        use_callback = False

    else:
        print(
            'Please insert your input directory of sample pool ' +
            '(like: C:/Users/Admin/Downloads/sample-pool/:).')
        MusicSampleConfiguration.sample_pool_path = input()
        print(
            'Please insert your output directory where your song should be saved ' +
            '(like: C:/Users/Admin/Downloads/output/:).')
        MusicSampleConfiguration.output_directory = input()

    MusicSampleStreamCreator.create_new_music_file()


if __name__ == "__main__":
    run()
