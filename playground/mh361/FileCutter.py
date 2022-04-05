from pydub import AudioSegment
import glob

# Length of the chopped file part in seconds
length = 5
music_dir = 'C:\\Users\\hennm\\Dropbox\\Studium\\Master\\Semester 2\\Tec Lab\\Music\\'

wav_filenames = glob.glob(str(music_dir + '*.wav*'))
wav_file = wav_filenames[0]

mp3_filenames = glob.glob(str(music_dir + '*.mp3*'))
mp3_file = mp3_filenames[0]

# WAV File
song = AudioSegment.from_wav(wav_file)

# MP3 File
#song = AudioSegment.from_mp3(mp3_file)

# Other File Formats
"""
mp4_version = AudioSegment.from_file("never_gonna_give_you_up.mp4", "mp4")
wma_version = AudioSegment.from_file("never_gonna_give_you_up.wma", "wma")
aac_version = AudioSegment.from_file("never_gonna_give_you_up.aiff", "aac")
"""

# Audio Slicing / second calculation / chunk definition
song_duration = song.duration_seconds
seconds = length * 1000
first_chunk = song[:seconds]
last_chunk = song[-seconds:]

# Save the file / chunks
#song.export("chunk.mp3", format="mp3")