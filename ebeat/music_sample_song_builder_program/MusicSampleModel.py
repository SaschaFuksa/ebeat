'''
Model to hold music file name to end and start edges
'''


class MusicSampleModel:

    def __init__(self, name: str, start: [], end: []):
        self.end = end
        self.start = start
        self.name = name
