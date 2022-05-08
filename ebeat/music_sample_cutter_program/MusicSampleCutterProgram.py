from ebeat.music_sample_cutter_program.MusicSampleCreator import MusicSampleCreator


# Run function to receive the user Inputs like Input directory / Output directory
def run():
    print('Please insert your input directory (like: C:/Users/Admin/Downloads/input/: With / at the end).')
    input_directory = input()
    print('Please insert your output directory (like: C:/Users/Admin/Downloads/output: without / at the end).')
    output_directory = input()
    print('Please insert your sample length in seconds. If you left 0, you can generate stochastic samples.')
    sample_length = int(input())
    # Stochastic approach where the user needs to enter details for silence detection
    if sample_length == 0:
        print(
            'Please insert amount of samples you want to create. If you left 0, all stochastic samples will be saved.')
        amount_of_samples = int(input())
        print(
            'Please insert minimum length of silence. If you left 0 or value smaller as 100 (0,1 seconds), default of '
            '100 will be used.')
        min_silence_length = int(input())
        if min_silence_length < 100:
            min_silence_length = 100
        print(
            'Please insert length of silence_threshold. If you left 0, default of -30 will be used.')
        silence_threshold = int(input())
        if silence_threshold == 0:
            silence_threshold = -30
        music_sample_creator = MusicSampleCreator.stochastic_cutting_data(input_directory, output_directory,
                                                                          amount_of_samples,
                                                                          min_silence_length, silence_threshold)
    else:
        music_sample_creator = MusicSampleCreator.equal_cutting_data(input_directory, output_directory, sample_length)
    music_sample_creator.create_samples()


if __name__ == "__main__":
    run()
