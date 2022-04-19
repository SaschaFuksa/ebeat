import os


class MusicFileCollector:
    """MusicFileCollector finds songs in the directory which the user enters as an Input directory.
It can handle wav or mp3 files."""

    def __init__(self, directory: str):
        self.directory = directory

    def find_songs(self):
        songs = []
        all_files = os.listdir(self.directory)
        for file in all_files:
            if file.endswith(".wav") or file.endswith(".mp3"):
                songs.append(file)
        return songs
