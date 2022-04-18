import librosa
import numpy as np


class SampleLoader:
    # load more files with librosa
    def load_samples(self):
        pathAudio = "C:/Users/treib/Documents/HDM/2. Semester/Technology Lab/Musik/Test/"
        files = librosa.util.find_files(pathAudio, ext=['wav'])
        files = np.asarray(files)
        for y in files:
            data, sr = librosa.load(y)

        print('data')
