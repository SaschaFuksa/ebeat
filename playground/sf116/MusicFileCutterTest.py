import unittest

from playground.sf116.EqualMusicFileCutter import EqualMusicFileCutter
from playground.sf116.IMusicFileCutter import IMusicFileCutter
from playground.sf116.RandomMusicFileCutter import RandomMusicFileCutter


class MyTestCase(unittest.TestCase):
    cutter:IMusicFileCutter

    def test_cut_in_equal(self):
        cutter = EqualMusicFileCutter()
        # testen ob von [Test-File] Samples gleich lang sind
        self.assertEqual(True, False)

    def test_cut_in_random(self):
        cutter = RandomMusicFileCutter()
        # testen ob von [Test-File] Samples verschieden lang sind
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
