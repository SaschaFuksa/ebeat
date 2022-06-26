from ebeat.music_sample_stream_creator_program.MusicSampleConfiguration import MusicSampleConfiguration
from ebeat.music_sample_stream_creator_program.MusicSampleStreamCreator import MusicSampleStreamCreator


def run():
    """
    Runs program and at first, insert relevant inputs like directories.
    """
    print('Please insert your input directory of train samples (like: C:/Users/Admin/Downloads/input/:).')
    MusicSampleConfiguration.train_sample_path = input()
    if MusicSampleConfiguration.train_sample_path == 'sf':
        MusicSampleConfiguration.train_sample_path = 'C:/Users/Admin/OneDrive/Dokumente/Studium/Technology Lab/Technology Lab Team 4/Techno Titel/train/samples-stochastic/'
        MusicSampleConfiguration.sample_pool_path = 'C:/Users/Admin/OneDrive/Dokumente/Studium/Technology Lab/Technology Lab Team 4/Techno Titel/music-pool/samples/all_samples/'
        MusicSampleConfiguration.output_directory = 'C:/Users/Admin/OneDrive/Dokumente/Studium/Technology Lab/Technology Lab Team 4/Techno Titel/music_out/'
        MusicSampleConfiguration.model_path = 'C:/Users/sasch/Downloads/model/weights-improvement-215-0.0084-bigger.hdf5'
        MusicSampleConfiguration.use_model = False
        MusicSampleConfiguration.new_song_name = 'Tester'
    elif MusicSampleConfiguration.input_directory == 'mh':
        MusicSampleConfiguration.input_directory = 'C:/Users/hennm/OneDrive/Technology Lab/Techno Titel/train/samples-stochastic/'
        MusicSampleConfiguration.output_directory = 'C:/Users/hennm/OneDrive/Technology Lab/Techno Titel/music-pool/samples/all_samples/'
        MusicSampleConfiguration.model_path = 'C:/Users/hennm/PycharmProjects/ebeat/playground/mh361/Challenge 3/Final Model/Final Model 300 Epochs 6691 Accuracy.h5'
        MusicSampleConfiguration.use_model = True
        MusicSampleConfiguration.new_song_name = 'Final'
    else:
        print(
            'Please insert your input directory of sample pool ' +
            '(like: C:/Users/Admin/Downloads/sample-pool/:).')
        MusicSampleConfiguration.sample_pool_path = input()
        print(
            'Please insert your output directory where your song should be saved ' +
            '(like: C:/Users/Admin/Downloads/output/:).')
        MusicSampleConfiguration.output_directory = input()
        print('Do you want to use existing model? (y/n).')
        use_model = input()
        if use_model == 'y':
            MusicSampleConfiguration.use_model = True
            print('Please insert your model (.hdf5) directory ' +
            '(like: C:/Users/Admin/Downloads/model/:).')
            MusicSampleConfiguration.model_path = input()
        print('Please insert name of new song (if left blank: Karacho)')
        new_name = input()
        if new_name != '':
            MusicSampleConfiguration.new_song_name = new_name
    MusicSampleStreamCreator.create_sample_stream()


if __name__ == "__main__":
    run()
