class MusicSampleConfiguration:
    """All configuration attributes for program.

    Attributes:
        train_sample_path           Path to training samples to use for model training in str.
        sample_pool_path            Path to sample pool to create song stream in str.
        edge_size                   Size of end and start edges of samples in int.
        batch_size                  Size of batches in int.
        epochs                      Amount of epochs to train model in int.
        use_callback                Flag to use callbacks (True)
        model_path                  Path to already existing model in str.
    """
    input_directory = ''
    output_directory = ''
    model_path = ''
    edge_size = 50
    batch_size = 2
    latent_dim = 1000
    epochs = 100
    use_callback = False
