import wave

import numpy
from matplotlib import pyplot as plt

import SamplingConstants

path = SamplingConstants.IN_SINGLE_SAMPLE_DIRECTORY + SamplingConstants.IN_SINGLE_FILE_NAME
signal_wave = wave.open(path, 'r')
sample_rate = signal_wave.getframerate()

sig = numpy.frombuffer(signal_wave.readframes(sample_rate), dtype=numpy.int16)

plt.figure(1)
plt.plot(sig)
plt.show()
