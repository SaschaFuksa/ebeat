import tensorflow
import pathlib
import glob
import pretty_midi
#import audio-to-midi

dname = 'C:\\Users\\hennm\\Dropbox\\Studium\\Master\\Semester 2\\Tec Lab\\Music\\'
filenames = glob.glob(str(dname+'*.mp3*'))
sample_file = filenames[1]
print(filenames)
#pm = pretty_midi.PrettyMIDI(sample_file)
