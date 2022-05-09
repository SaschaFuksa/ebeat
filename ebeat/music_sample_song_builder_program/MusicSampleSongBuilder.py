'''
Build song on given suggestion of lstm model by predicted sample names
'''
from pydub import AudioSegment


class MusicSampleSongBuilder():
    def __init__(self, input_directory: str, output_directory: str):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def save_song(self, sample_order):
        new_song = AudioSegment.empty()
        for sample in sample_order:
            complete_path = self.input_directory + sample
            sample = AudioSegment.from_wav(complete_path)
            new_song += sample
        new_song.export(self.output_directory + 'karacho.wav', format='wav')
