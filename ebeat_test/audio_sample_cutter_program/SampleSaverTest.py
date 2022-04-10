import os
import unittest

import numpy

from ebeat.music_sample_cutter_program.SampleSaver import SampleSaver


class MyTestCase(unittest.TestCase):
    test_folder = 'sample_saver_test_data'
    audio_data = numpy.array([1, 2, 3, 4, 5])

    def test_something(self):
        sample_saver = SampleSaver(self.test_folder + '/', 'test', 'wav', 44100)
        audios = [self.audio_data, self.audio_data, self.audio_data]
        for audio in audios:
            sample_saver.save_sample(audio)
        files = os.listdir(self.test_folder)
        files.remove('.gitkeep')
        self.assertEqual(len(files), 3)
        i = 1
        for file in files:
            self.assertEqual(file, 'test_' + str(i) + '.wav')
            i += 1

    def tearDown(self):
        files = os.listdir(self.test_folder)
        files.remove('.gitkeep')
        for file in files:
            os.remove(self.test_folder + '/' + file)


if __name__ == '__main__':
    unittest.main()
