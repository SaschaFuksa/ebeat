from abc import ABC, abstractmethod

"""
MusicSampleCutter is used as a signature of the method without implementing it.
The abstract method will be overridden in the subclass EqualMusicSampleCutter or StochasticMusicSampleCutter.
"""


class MusicSampleCutter(ABC):

    @abstractmethod
    def cut_music_file(self, music_file_path: str) -> []:
        pass
