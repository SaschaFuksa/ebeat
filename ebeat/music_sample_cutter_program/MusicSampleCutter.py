from abc import ABC, abstractmethod


class MusicSampleCutter(ABC):

    @abstractmethod
    def cut_music_file(self, music_file_path: str) -> []:
        pass
