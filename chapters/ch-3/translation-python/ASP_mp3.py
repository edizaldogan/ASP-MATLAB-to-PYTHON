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
print(input_signal)
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

'''
Applying direct downsampling-upsampling by 2 results in replacing every
other sample by zero. The result is that an artifact signal is added,
whose spectrum is the quadrature mirror image of that of the chirp. Two
sinusoids can be heard at any time. (Notice we multiply the signal by 2
when upsampling, so that the power of the signal remains unchanged.)
'''

downsampled = input_signal[::2]
print(downsampled)
upsampled = np.zeros(2*len(downsampled))
upsampled[::2]=2*downsampled
print(upsampled)
plot_spectrogram(upsampled,1024,Fs,256)
'''
Zeroing out half of the elements causes to lose half of the power of 
the signal. That's why we multiply by two before upsampling. 
Multiplying by two compensates the loss.
Not multiplying by two causes a 6dB change on the spectrogram. The
change can be observed by realising all the points moving upwards 
(going above the red level). It pushes more points to pass the redness 
threshold.
By doing the above conversions we transform the input signal from the 
form: [a,b,c,d,e,...] to the form [2a,0,2c,0,2e,...]
The aim of doing this is to intentionally show what happens if no
anti-aliasing filtering applied before sampling. We deliberately broke
the signal to show the effect of missing proper filtering.

Notice that we multiplied the input signal s[n]=[a,b,c,d,e,...] by the 
array x = [2,0,2,0,2,...] whose elements are basically x[n] = 1 + cos(pi*n).
s_new = s[n]+x[n] = s[n]+s[n]cos(pi*n) is the new signal.
s[n] has a frequency spectrum that sweeps from 0Hz to 4000Hz.
s[n]cos(pi*n)= has a frequency spectrum that sweeps from 4000Hz to 0Hz.
w = 2*pi*f/fs = 2*pi*f/8000 = pi which yields f = 4000Hz.
Multiplying by a cosine wave with frequency 4000Hz in time domain corresponds
to shifting an the spectrum by an amount of 4000Hz in the frequency domain.
So, the original chirp sweeps from 0 to 4000Hz while an image sweeps from
0 to -4000Hz. Adding 4000 to this image we realize that it appears as
sweeping from 4000Hz to 0Hz. That's why we observe a second red line on
the opposite diagonal.
'''
# A audio player will be implemented here.

# %%
'''
Adding a quarter-band filter after upsampling eliminates the image
distortion, making sure that only one 
sinusoid appears at any time. During the first half of the chirp, that
sinusoid is the chirp itself; during the second half, the sinusoid is an
alias, due to the quadrature mirror image of the chirp. 
We design the synthesis filter as symmetric FIR, so as to avoid phase 
distortion, and we set its order to 1000, so as to obtain high stop-band
rejection (close to 80 dB). Notice the first 1000 samples are a transient.
'''
# b = fir1(n,Wn) uses a Hamming window to design an nth-order lowpass, 
# bandpass, or multiband FIR filter with linear phase. The filter type 
# depends on the number of elements of Wn.

G0 = signal.firwin(1000, 1/2)
W, H = signal.freqz(G0, 1, worN=512)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(W/np.pi, 20*np.log10(np.abs(H)))
plt.title('Magnitude')
plt.xlabel('Normalized Frequency ($\\times \\pi$ rad/sample)')
plt.ylabel('Magnitude (dB)')
plt.yticks(np.arange(-160,20,20))
plt.xticks(np.arange(0,1.1,0.1))
plt.grid(True)

plt.subplot(2, 1, 2)
# Calculate phase in radians, unwrap to remove 2*pi discontinuities, 
# convert to degrees.
phase = np.unwrap(np.angle(H))*(180 / np.pi)
plt.plot(W/np.pi, phase)
plt.title('Phase')
plt.xlabel('Normalized Frequency ($\\times \\pi$ rad/sample)')
plt.ylabel('Phase (degrees)')
plt.yticks(np.arange(-50000,0,10000))
plt.xticks(np.arange(0,1.1,0.1))
plt.grid(True)

plt.tight_layout()

# %%

G0_output = signal.lfilter(G0,1,upsampled)
# NB: The first 1000 samples are a filter transient reponse
plot_spectrogram(G0_output[1000:],1024,Fs,512)

# A audio player will be implemented here.

# %%

