import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from scipy.io.wavfile import write

temp_folder = 'C:\\Users\\Admin\\OneDrive\\Dokumente\\Studium\\Technology Lab\\Techno Titel\\'
wav_file = 'Tea K Pea - nauticals.wav'

rate, audData = scipy.io.wavfile.read(temp_folder + wav_file)

print("The sampling rate (i.e. data points per second) is " + str(rate))
print("The type of the data stored in each datum is " + str(audData.dtype))
print("The total number of data points is " + str(audData.shape[0]))
print("The number of channels (i.e. is it mono or stereo) is " + str(audData.shape[1]))
print("The wav length is " + str(audData.shape[0] / rate) + " seconds")

channel1 = audData[:, 0]
channel2 = audData[:, 1]

N = audData.shape[0]

time = np.arange(0, float(audData.shape[0]), 1) / rate

plt.figure()
plt.plot(channel2)
# plt.show()

samples = []
time = len(audData) / rate
amount_of_samples = time / 10
sample_size = int(13 * rate)
if amount_of_samples >= 1:
    i = 0
    max_len = len(audData)
    while i < max_len:
        sample = audData[i:i + sample_size]
        samples.append(sample)
        i += sample_size
print(len(samples))
i = 1
for sample in samples:
    write(temp_folder + str(i) + "example.wav", rate, sample.astype(np.int16))
    i += 1

# plt.figure(1)
# plt.subplot(211)
# plt.plot(time, channel1, linewidth=0.01, alpha=0.7, color='#ff7f00')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.subplot(212)
# plt.plot(time, channel2, linewidth=0.01, alpha=0.7, color='#ff7f00')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()
