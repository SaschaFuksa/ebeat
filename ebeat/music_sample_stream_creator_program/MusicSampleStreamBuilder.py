from pydub import AudioSegment

from ebeat.music_sample_stream_creator_program.MusicSampleConfiguration import MusicSampleConfiguration


class MusicSampleStreamBuilder:
    def __init__(self):
        pass

    @staticmethod
    def save_song(selected_samples):
        new_song = AudioSegment.empty()
        for sample in selected_samples:
            complete_path = MusicSampleConfiguration.sample_pool_path + sample + '.mp3'
            sample = AudioSegment.from_mp3(complete_path)
            new_song += sample
        new_song.export(MusicSampleConfiguration.output_directory + 'karacho.mp3', format='.mp3')