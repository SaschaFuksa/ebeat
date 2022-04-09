import unittest

from ebeat.music_sample_cutter_program.EqualMusicSampleCutter import EqualMusicSampleCutter


class EqualMusicSampleCutterTest(unittest.TestCase):
    path = 'equal_cutter_test_data'
    predefined_length = 2
    expected_amount_of_samples = 7
    rate = 44100

    def test_cut_equal_sized_file(self):
        file = '/test_file.wav'
        cutter = EqualMusicSampleCutter(self.predefined_length)
        samples = cutter.cut_music_file(self.path + file)
        self.assertEqual(len(samples), self.expected_amount_of_samples)
        sample_length = len(samples[0]) / self.rate
        self.assertEqual(sample_length, self.predefined_length)


if __name__ == '__main__':
    unittest.main()
