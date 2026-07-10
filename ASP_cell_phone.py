# %%

# Chapter 1 - How is speech processed in a cell phone conversation?
# This is a companion file to the book "Applied Signal Processing",
# by T. Dutoit and F. Marques, Springer 2008.
#
# In this script, we will see how LPC-based analysis-synthesis lies at the
# very heart of mobile phone transmission of speech. We will first examine
# the contents of a speech file, in Section 1. Then we will perform LP
# analysis and synthesis on a voice and on an unvoiced frame, in Sections 2
# and 3 respectively. We will then generalize this approach to the complete
# speech file, by first synthesizing all frames as voiced and imposing a
# constant pitch, in Section 4, then by synthesizing all frames as unvoiced
# in section 5, and finally by using the original pitch and voicing
# information, in Section 6.
#
# Copyright T. Dutoit, N. Moreau, 2008
#
# Python translation by Ediz Aldogan.

# %%

# 1. Examining a speech file
# Let us load file 'speech.wav', listen to it, and plot its samples. This
# file contains the sentence "Paint the circuits" sampled at 8 kHz, with 16
# bits. (This sentence was taken from the Open Speech Repository on the
# web)

import numpy             as np
import matplotlib.pyplot as plt
from scipy.io            import wavfile
sample_rate, audio   = wavfile.read(r"speech.wav")
total_duration       = len(audio)
time                 = np.arange(0, total_duration, 1)
plt.figure  (figsize=(10, 4))
plt.title   ("Speech Signal")
plt.xlabel  ("Time [samples]")
plt.ylabel  ("Amplitude")
plt.plot    (time, audio)
plt.grid    (True)
plt.show    ()

# %%

# The file is about 1.1 s long (9000 samples). One can easily spot the
# position of the four vowels in this plot, since vowels usually have
# higher amplitude than other sounds. The vowel 'e' in "the", for instance,
# is approximately centered on sample 3500.

# %%

