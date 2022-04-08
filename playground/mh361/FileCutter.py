from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import make_chunks
import glob
import os
import IPython.display as display
from pydub.playback import play

# Length of the chopped file part in seconds
size = 6

# Music file directory
music_dir = 'C:\\Users\\hennm\\Dropbox\\Studium\\Master\\Semester 2\\Tec Lab\\Music\\'

# Get the files within the directory
wav_filenames = glob.glob(str(music_dir + '*.wav*'))
wav_file = wav_filenames[0]
mp3_filenames = glob.glob(str(music_dir + '*.mp3*'))
mp3_file = mp3_filenames[0]

# File Pre-Fix name
file_prefix = os.path.basename(wav_file).split('.')[0]

# WAV File
song = AudioSegment.from_wav(wav_file)

# Song duration / time to milliseconds/ chunk definition
song_duration = song.duration_seconds * 1000
chunk_size_ms = size * 1000

amount_of_chunks = song_duration / chunk_size_ms

# Get first and last chunk -> testing
first_chunk = song[:chunk_size_ms]
last_chunk = song[-chunk_size_ms:]

# split audio file based on fix size - seconds
chunks = make_chunks(song, chunk_size_ms)
# Loop over fixed chunks and export chunks as wav files
for i, chunk in enumerate(chunks):
    chunk_name = file_prefix + " chunk{0}".format(i)  # set the file name + chunk number
    print(chunk_name)
    #chunk.export(chunk_name, format="wav") #export the chunks as wav file

# split audio file based on silence
silence_chunks = split_on_silence(song, min_silence_len=400, silence_thresh=-40)
# Loop over silence_chunks and export them
for i, silence in enumerate(silence_chunks):
    s_chunk_name = file_prefix + " chunk{0}".format(i)  # set the file name + chunk number
    print(silence.duration_seconds)
    #silence.export(s_chunk_name, format="wav") #export the chunks as wav file

# Play chunk file
#play(chunks[0])

# Play chunk in Audio-Widget
#display.Audio(wav_file, rate=22000, autoplay=False)


# Save the file / chunks
#song.export("chunk.mp3", format="mp3")