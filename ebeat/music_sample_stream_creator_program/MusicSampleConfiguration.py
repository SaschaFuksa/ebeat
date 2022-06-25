class MusicSampleConfiguration:
    """All configuration attributes for program.

    Attributes:
        train_sample_path           Path to training samples to use for model training in str.
        sample_pool_path            Path to sample pool to create song stream in str.
        output_directory            Path to save new song stream in str.
        model_path                  Path to already existing model in str.
        use_model                   True if use model, False if don't use model and build new
        new_song_name               Name of new song
    """
    train_sample_path = ''
    sample_pool_path = ''
    output_directory = ''
    model_path = ''
    use_model = False
    new_song_name = 'karacho'
