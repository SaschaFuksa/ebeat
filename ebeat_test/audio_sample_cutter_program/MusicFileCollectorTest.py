import unittest

from ebeat.music_sample_cutter_program.MusicFileCollector import MusicFileCollector


class MusicFileCollectorTest(unittest.TestCase):
    path = "test_data"

    def test_find_songs(self):
        finder = MusicFileCollector(self.path)
        songs = finder.find_songs()
        self.assertEqual(len(songs), 4)

    def test_list(self):
        test_list = ["test_a.mp3", "test_a.wav", "test_b.mp3", "test_b.wav"]
        finder = MusicFileCollector(self.path)
        songs = finder.find_songs()
        self.assertListEqual(test_list, songs, "Lists are equal")


if __name__ == '__main__':
    unittest.main()
