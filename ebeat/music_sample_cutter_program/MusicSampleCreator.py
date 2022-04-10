from ebeat.music_sample_cutter_program.EqualMusicSampleCutter import EqualMusicSampleCutter
from ebeat.music_sample_cutter_program.MusicFileCollector import MusicFileCollector
from ebeat.music_sample_cutter_program.MusicSampleCutter import MusicSampleCutter
from ebeat.music_sample_cutter_program.SampleSaver import SampleSaver


class MusicSampleCreator:
    music_sample_cutter: MusicSampleCutter

    def __init__(self, input_directory: str, output_directory: str, sample_length: int, amount_of_samples: int):
        self.amount_of_samples = amount_of_samples
        self.sample_length = sample_length
        self.output_directory = output_directory
        self.input_directory = input_directory

    def create_samples(self):
        file_collector = MusicFileCollector(self.input_directory)
        songs = file_collector.find_songs()
        rate = 44100

        if self.sample_length > 0:
            self.music_sample_cutter = EqualMusicSampleCutter(self.sample_length)
        else:
            self.music_sample_cutter = StochasticMusicSampleCutter

        for song in songs:
            samples = self.music_sample_cutter.cut_music_file(song)
            # " Gänsefüßchen stehen für die drei Variablen in der
            # Sample Saver Klasse (self.track name, audio type, self rate)
            sample_saver = SampleSaver(self.output_directory, "", "", rate)

            for sample in samples:
                sample_saver.save_sample(sample)
