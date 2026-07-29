# %%
'''
Chapter 3 - How is sound processed in an MP3 player?
This is a companion file to the book "Applied Signal Processing", 
by T.Dutoit and F. Marques, Springer 2008.
 
It is supposed to be run cell-by-cell, using the cell mode of 
MATLAB 6 and later versions. Search for "what are cells" in the 
product help of MATLAB to know how to use them.
 
This file uses the SIGNAL_PROCESSING toolbox of MATLAB.
'''
# %%
'''
In this script, we will see how the limitations of the human auditory
process have been taken into account for the design of perceptual audio
coders, with special emphasis on the principles underlying the MPEG-1
Layer-I audio coding norm. We start by examining a two-channel filter bank
using conventional filters (Section 1) and QMF filters (Section 2). We
then extend our study to a 32-channel PQMF filter bank (Section 3) and
show how it can be efficiently implemented using block-based lapped
transforms (Section 4). We conclude by providing our filter bank with
a perceptual quantizer for sub-band signals (Section 5).

Copyright N. Moreau, T. Dutoit (2007)
Python translation by Ediz Aldogan.
'''
import matplotlib.pyplot as plt
# Set global figure parameter 
# This makes all the background colors of figures white by default.
# (MATLAB equivalent: set(0,'defaultFigureColor','w'))
plt.rcParams['figure.facecolor'] = 'white'

# %%
'''
1. Two-channel filter bank
The MPEG 1 Layer I coder is based on quantifying the sub-band signals of
an analysis-synthesis filter bank (as we will see later, this can also be
interpreted in terms of block-transform-based processing).  
The analysis filter-bank at the encoder splits the signal into (ideally,
non-overlapping) frequency bands. The resulting narrow-band signals are
then decimated, and quantized. The decoder upsamples each sub-band signal
and performs band-pass-filtering so as to eliminate the aliasing images
due to upsampling.    
 
Before examining a complete M-channel filter bank, we first discuss 
the 2-channel case, and the effect of a decimation and interpolation in
each branch. 

Let us first generate a 4-seconds chirp signal (from 0 to 4 kHz) with a
sampling rate of 8 kHz, which we will use as a reference throughout
this Section.  
'''
from scipy import signal
import numpy as np

Fs = 8000
# chirp(t0,f0,t1,f1) is a frequency-swept cosine generator.
t   = np.arange(0,4*Fs)/Fs  # Times at which to evaluate the waveform.
f0  = 0                     # Frequency (e.g. Hz) at time t=0.
t1  = 4                     # Time at which f1 is specified.
f1  = 4000                  # Frequency (e.g. Hz) of the waveform at time t1.
input_signal = signal.chirp(t,f0,t1,f1)

# A audio player will be implemented here.

# %%
'''
argument order in old and new versions of spectrogram function: 
OLD: [S, F, T] = specgram(x, nfft, fs, window, noverlap)
NEW: [S, F, T] = spectrogram(x,win,nOverlap,freqSpec,Fs)
'''
# Notice we write detrend=False beacuse default value of detrend was 
# detrend=constant which causes a DC component to appear near the 0Hz 
# level. The bottom line shows up in red which is wrong.
def plot_spectrogram(audio, nfft, fs, window):
    f, t, Sxx = signal.spectrogram(audio, 
                                   fs=fs,
                                   window=np.hanning(window),
                                   nperseg=window,
                                   noverlap=window/2,
                                   nfft=nfft,
                                   detrend=False,
                                   mode='magnitude')
    plt.figure      (figsize=(10, 8))
    # Python uses 'viridis' color map as default. 
    # Instead we will use MATLAB's 'jet' colormap for exact visual match
    plt.pcolormesh  (t, f, 20*np.log10(Sxx), shading='auto', cmap='jet')
    plt.title       ("Spectrogram")
    plt.ylabel      ('Frequency (Hz)')
    plt.xlabel      ('Time (s)')

plot_spectrogram(input_signal,1024,Fs,256)
# %%

