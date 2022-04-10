import unittest

from ebeat.music_sample_cutter_program.StochasticMusicSampleCutter import StochasticMusicSampleCutter


class StochasticMusicSampleCutterTest(unittest.TestCase):
    path = 'equal_cutter_test_data'

    def test_cut_unequal_sized_file(self):
        file = '/test_file.wav'
        cutter = StochasticMusicSampleCutter(0, 400, -40)
        samples = cutter.cut_music_file(self.path + file)
        self.assertEqual(len(samples), 2)  # add assertion here


if __name__ == '__main__':
    unittest.main()
