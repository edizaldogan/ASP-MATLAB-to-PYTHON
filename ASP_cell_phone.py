# %%
'''
Chapter 1 - How is speech processed in a cell phone conversation?
This is a companion file to the book "Applied Signal Processing",
by T. Dutoit and F. Marques, Springer 2008.

In this script, we will see how LPC-based analysis-synthesis lies at the
very heart of mobile phone transmission of speech. We will first examine
the contents of a speech file, in Section 1. Then we will perform LP
analysis and synthesis on a voice and on an unvoiced frame, in Sections 2
and 3 respectively. We will then generalize this approach to the complete
speech file, by first synthesizing all frames as voiced and imposing a
constant pitch, in Section 4, then by synthesizing all frames as unvoiced
in section 5, and finally by using the original pitch and voicing
information, in Section 6.

Copyright T. Dutoit, N. Moreau, 2008

Python translation by Ediz Aldogan.
'''

# Set global figure parameter 
# (MATLAB equivalent: set(0,'defaultFigureColor','w'))
plt.rcParams['figure.facecolor'] = 'white'

# %%
'''
1. Examining a speech file
Let us load file 'speech.wav', listen to it, and plot its samples. This
file contains the sentence "Paint the circuits" sampled at 8 kHz, with 16
bits. (This sentence was taken from the Open Speech Repository on the
web)
'''

# MATLAB normalizes the value between the range -1 and 1 with the function audioread.
# However, Python reads the raw format which ranges between -32768 and 32767
# To normalize we use the following equation: audio = audio_raw / 32768
import numpy             as np
import matplotlib.pyplot as plt
from scipy.io            import wavfile
sample_rate, audio_raw   = wavfile.read("speech.wav")
audio = audio_raw / 32768
total_duration       = len(audio)
speech_time          = np.arange(0, total_duration, 1)
y_tick_marks         = np.arange(-1.0, 1.1, 0.1)
x_tick_marks         = np.arange(0,10000, 2000)
plt.figure  (figsize=(10, 8))
plt.title   ("Speech Signal")
plt.yticks  (y_tick_marks)
plt.xticks  (x_tick_marks)
plt.xlabel  ("Time [samples]")
plt.ylabel  ("Amplitude")
plt.plot    (speech_time, audio)
plt.grid    (True)
plt.show    ()

# %%
'''
The file is about 1.1 s long (9000 samples). One can easily spot the
position of the four vowels in this plot, since vowels usually have
higher amplitude than other sounds. The vowel 'e' in "the", for instance,
is approximately centered on sample 3500.
'''

# %%
'''
As such, however, the speech waveform is not "readable", even by an
expert phonetician. Its information (phonetic) content is hidden. In
order to reveal it to the eyes, let us plot a spectrogram of the signal.
For a better graphical result, we choose a wideband spectrogram, by
imposing the length of each frame to be approximately 5 ms long (40
samples) and a hamming weighting window.
'''

from scipy import signal

def plot_spectrogram(audio, fs=8000):
    """
    Replicates MATLAB's specgram(speech, 512, 8000, hamming(40)) function.
    Uses a 40-sample Hamming window for formant visualization.
    """
    f, t, Sxx = signal.spectrogram(audio, 
                                   fs=fs,
                                   window=np.hamming(40),
                                   nperseg=40,
                                   noverlap=20,
                                   nfft=512)
    plt.figure      (figsize=(10, 8))
    # Python uses 'viridis' color map as default. 
    # Instead we will use MATLAB's 'jet' colormap for exact visual match
    plt.pcolormesh  (t, f, 10 * np.log10(Sxx), shading='auto', cmap='jet')
    plt.title       ("Spectrogram")
    plt.ylabel      ('Frequency (Hz)')
    plt.xlabel      ('Time (s)')
    plt.show        ()

plot_spectrogram(audio)

# %%
'''
In this plot, pitch periods appear as vertical lines. As a matter of
fact, since the length of analysis frames is very small, some frames fall
on the peaks (resp., on the valleys) of pitch periods, and thus appear as
a darker (resp., lighter) vertical lines.

In contrast, formants (resonant frequencies of the vocal tract) appear as
dark (and rather wide) horizontal traces. Although their frequency is not
easy to measure with precision, experts looking at such a spectrogram can
actually often read it (i.e. guess the corresponding words). This clearly
shows that formants are a good indicator of the underlying speech sounds.
'''

# %%
'''
2. Linear prediction synthesis of 30 ms of voiced speech
Let us extract a 30 ms frame from a voiced part (i.e. 240 samples) of the
speech file, and plot its samples. 
'''

# The following interval corresponds to the time interval of letter 'e' 
# from the speech file.
input_frame_for_letter_e = audio[3499:3739]
time = np.arange(0,240,1)
plt.figure  (figsize=(10, 8))
plt.plot    (input_frame_for_letter_e)
plt.title   ('Zooming on the letter "e"')
plt.xlabel  ('Time (samples)')
plt.ylabel  ('Amplitude')
plt.grid    (True)
plt.show    ()

# %%
'''
As expected this sound is approximately periodic (period=65 samples, i.e.
8 ms; fundamental frequency = 125 Hz). Notice, though, that this is only
apparent; in practice, no sequence of samples can be found more than once
in the frame.
'''

# %%
'''
Now let us see the spectral content of this speech frame, by plotting its
periodogram on 512 points (using a normalized frequency axis; remember 
pi corresponds to Fs/2, i.e. to 4000 Hz here). 
'''

def plot_periodogram(frame):
    """
    Replicates MATLAB's periodogram(input_frame, [], 512) function 
    with normalized amplitude and frequency settings.

    For Python perodogram is a 2D plot with normalized frequency on the 
    horizontal axis and the PSD on the vertical axis. 
    It is in the following format:
    frequency, PSD = periodogram(x, fs, window=None,nfft=integer)
    The window is chosen to be rectangular by default if it is left blank. 
    P stands for power and xx for autocorrelation.
    detrend is fixed as a constant value, we need to set it as False.
    Otherwise, it subtracts the DC component of the signal (removes linear 
    or constant trends) before taking the Fourier transform.
    """

    # fs=2*pi and detrend=False mimic MATLAB's normalized PSD assumptions
    f_rad, Pxx = signal.periodogram(frame, 
                                    fs=2*np.pi, 
                                    window='boxcar', 
                                    detrend=False,
                                    nfft=512)
    
    # Map the frequency axis between 0 and 1 (as multiples of pi)
    f_normalized = f_rad / np.pi
    
    # Convert the linear PSD to Decibels (dB)
    Pxx_db = 10 * np.log10(Pxx)
    
    plt.figure  (figsize=(10, 8))
    plt.plot    (f_normalized, Pxx_db)
    plt.title   ('Periodogram Power Spectral Density Estimate')
    plt.ylabel  ('Power/frequency (dB/(rad/sample))')
    plt.xlabel  ('Normalized Frequency ($\\times \\pi$ rad/sample)')
    plt.xlim    (0, 1)
    plt.grid    (True)
    plt.show    ()

plot_periodogram(input_frame_for_letter_e)

# %%
