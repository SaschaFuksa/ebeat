import unittest

from ebeat.music_sample_cutter_program.EqualMusicSampleCutter import EqualMusicSampleCutter


class EqualMusicSampleCutterTest(unittest.TestCase):
    path = 'equal_cutter_test_data'

    def test_cut_wav_file(self):
        file = '/test_file.wav'
        cutter = EqualMusicSampleCutter(2)
        samples = cutter.cut_music_file(self.path + file)
        self.assertEqual(len(samples), 7)
        sample_length = len(samples[0]) / 44100
        self.assertEqual(sample_length, 2)


if __name__ == '__main__':
    unittest.main()
