
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import spectrogram
import snr
import PQMF32_prototype
import MPEG1_psycho_acoustic_model1 as mpeg_pam
import MPEG1_bit_allocation as mpeg_ba
import sounddevice as sd

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
Fs = 8000
# chirp(t0,f0,t1,f1) is a frequency-swept cosine generator.
t   = np.arange(0,4*Fs)/Fs  # Times at which to evaluate the waveform.
f0  = 0                     # Frequency (e.g. Hz) at time t=0.
t1  = 4                     # Time at which f1 is specified.
f1  = 4000                  # Frequency (e.g. Hz) of the waveform at time t1.
input_signal = signal.chirp(t,f0,t1,f1)
sd.play(input_signal,Fs)

# %%
spectrogram.plot_spectrogram(input_signal,1024,Fs,256)
# %%

'''
Applying direct downsampling-upsampling by 2 results in replacing every
other sample by zero. The result is that an artifact signal is added,
whose spectrum is the quadrature mirror image of that of the chirp. Two
sinusoids can be heard at any time. (Notice we multiply the signal by 2
when upsampling, so that the power of the signal remains unchanged.)
'''

downsampled = input_signal[::2]
upsampled = np.zeros(2*len(downsampled))
upsampled[::2]=2*downsampled
spectrogram.plot_spectrogram(upsampled,1024,Fs,256)
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
sd.play(upsampled,Fs)

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

G0 = signal.firwin(1001, 1/2)
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
spectrogram.plot_spectrogram(G0_output[1000:],1024,Fs,512)
sd.play(G0_output[1000:],Fs)

# %%
'''
Adding another quarter-band filter before downsampling removes the
aliasing distortion, i.e. the alias in the second half of the chirp. This synthesis 
filter can a priori be identical to the analysis filter.
'''

H0  = G0
H0_output = signal.lfilter(H0,1,input_signal)
downsampled = H0_output[::2]
upsampled = np.zeros(2*len(downsampled))
upsampled[::2] = 2*downsampled
G0_output = signal.lfilter(G0,1,upsampled)

# NB: The first 2000 samples are a filter transient reponse
spectrogram.plot_spectrogram(G0_output[2000:],1024,Fs,256)
sd.play(G0_output[2000:],Fs)

# %%

'''
Using HF quarter-band filters instead of LF ones selects the second half
of the chirp.
'''
G1 = signal.firwin(1001, 1/2, pass_zero='highpass')
H1 = G1
H1_output = signal.lfilter(H1,1,input_signal)
downsampled = H1_output[::2]
upsampled = np.zeros(2 * len(downsampled))
upsampled[::2]=2*downsampled
G1_output = signal.lfilter(G1,1,upsampled)

spectrogram.plot_spectrogram(G1_output[2000:],1024,Fs,256)
sd.play(G1_output[2000:],Fs)

# %%

'''
We now examine the output of the 2-channel sub-band filter bank, by
adding  the two signals obtained above.
'''

synt_signal = G0_output + G1_output
spectrogram.plot_spectrogram(synt_signal[2000:],1024,Fs,256)
sd.play(synt_signal[2000:],Fs)

# %%
'''
Perfect reconstruction is not achieved, as shown in the previous
spectrogram. Closer examination of the error waveform shows that most of
the error lies in the center of the signal (this is not shown in the
spectrogram), i.e., for frequencies for which the H0 and H1 filters
overlap: more important aliasing cannot be avoided in each band for these
frequencies. 

Shift synt_signal to account for the filters delay. A symmetric FIR
filter of order N (length N+1) brings a delay of N/2 samples.
'''
error = synt_signal[1000:]-input_signal[:len(synt_signal)-1000]
plt.figure(figsize=(10, 8))
plt.plot(error)
plt.title("Reconstruction Error")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
sd.play(error,Fs)

# %%

'''
2. Two channel QMF filter bank
Perfect reconstruction is possible even with non ideal filters (i.e. even
with some aliasing in each band) provided the overall aliasing is canceled
when adding the low-pass and high-pass sub-band signals.
This, as we shall see, is mainly the responsibility of the analysis and
synthesis filters. 

(Johnston's type) QMF filters provide a solution for nearly perfect
reconstruction. In this example, we use Johnston's "B-12" QMF. 
'''
H0_QMF=[-0.006443977, 0.02745539, -0.00758164, -0.0913825,  0.09808522, 0.4807962]
H0_QMF_reverse=H0_QMF[::-1]
H0_QMF_extended=H0_QMF + H0_QMF_reverse
W,H0=signal.freqz(H0_QMF_extended,1, worN=512)

vector = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
H1_QMF=np.multiply(H0_QMF_extended, vector)
W,H1=signal.freqz(H1_QMF,1, worN=512)

plt.figure(figsize=(10,8))
plt.plot(W/np.pi,20*np.log10(abs(H0)),color='blue',label='H0(f)')
plt.plot(W/np.pi,20*np.log10(abs(H1)),'--',color='orange',label='H1(f)')
plt.legend()
plt.xlabel('Normalized frequency (*pi rad/sample)')
plt.ylabel('Magnitude (dB)')

# %%

'''
These filters are not very frequency selective (because their order is
low), and will therefore allow important aliasing in each band.
'''
Fs=8000
t= np.arange(0,4*Fs)/Fs  # Times at which to evaluate the waveform.
input_signal=signal.chirp(t,0,4,4000)
# LF band
H0_output=signal.lfilter(H0_QMF_extended,1,input_signal)
subband_0=H0_output[::2]
upsampled = np.zeros(2*len(subband_0))
upsampled[::2]=2*subband_0
G0_QMF=H0_QMF_extended
G0_output=signal.lfilter(G0_QMF,1,upsampled)
spectrogram.plot_spectrogram(G0_output,1024,Fs,256)
#sd.play(G0_output,Fs)

# %%
# HF band

H1_output=signal.lfilter(H1_QMF,1,input_signal)
subband_1=H1_output[::2]
upsampled = np.zeros(2*len(subband_0))
upsampled[::2]=2*subband_1
G1_QMF=-H1_QMF
G1_output=signal.lfilter(G1_QMF,1,upsampled)

spectrogram.plot_spectrogram(G1_output,1024,Fs,256)
#sd.play(G1_output,Fs)

# %%
'''
Perfect reconstruction is now achieved, because the QMF analysis and
synthesis filters are such that the aliasing in each band sum up to zero
(more precisely, close to 0). 
'''
synt_signal=G0_output+G1_output
spectrogram.plot_spectrogram(synt_signal,1024,Fs,256)
#sd.play(synt_signal,Fs)

# %%
'''
3. 32-Channel Pseudo-QMF filter bank
We now build a 32-channel PQMF filter bank, as implemented in the MPEG-1
Layer-I norm, and check its perfect reconstruction capability.

*MATLAB function involved:*
 
* |hn = PQMF32_prototype| returns in |hn| the impulse response of the prototype
low-pass symmetric filter of length 512 for building a 32-channel PQMF
filter bank. This filter is used in the MPEG-1 Layer-I coder. Its
normalized bandpass is 1/64 Hz and it satisfies the PR condition.  
'''
# Load the prototype lowpass filter
hn = PQMF32_prototype.PQMF32_prototype()

# %%
# Build 32 cosine modulated filters centered on normalized frequencies
# Fi=(2*i+1)/64 *1/2
PQMF32_Gfilters = np.zeros((32, 512))
for i in range(32):
    t2 = ((2*i+1)*np.pi/(2*32))*(np.arange(512)+16)
    PQMF32_Gfilters[i,:] = hn*np.cos(t2)

PQMF32_Hfilters=PQMF32_Gfilters[:, ::-1]

plt.figure(figsize=(10, 8))
for i in range(32):
    W,H=signal.freqz(PQMF32_Hfilters[i,:],1,worN=512,fs=44100)
    plt.plot(W,20*np.log10(abs(H)))
plt.title("32-Channel Pseudo-QMF Filter Bank")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.xlim(0, 22050)
plt.ylim(-100, 5) # Optional: Added to make the plot cleaner, mirroring standard DSP views
plt.grid(True)

# %%
'''
Clearly, PQMF filters are not ideal band-pass filters with normalized
bandwidth=1/64. But they almost satisfy the "pseudo" feature, which
assumes that the filter response in a given band only overlaps with its 2
neighboring bands. The total normalized bandwidth of each filter is 
indeed about  1/32. Notice the filter gain is 15 dB, i.e. 20*log10(32)/2.
As a result, passing twice through the filter produces a gain of 32,
which compensates for the decimation by 32. In the next lines, it is thus
not necessary to multiply upsampled sub-band signals by 32.

Let us now check the output of the PQMF filter bank when fed with 2
seconds of violin monophonic signal sampled at 44.100 Hz.
'''
sample_rate, audio_raw = wavfile.read('../audio_samples/violin.wav')
audio = audio_raw / 32768
total_duration = len(audio)
spectrogram.plot_spectrogram(audio,1024,Fs,256);
sd.play(audio,Fs)

# %%
output_signal=np.zeros(len(audio))
for i in range(32):
    Hi_output=signal.lfilter(PQMF32_Hfilters[i,:],1,audio)
    subband_i=Hi_output[::32]
    upsampled_i = np.zeros(len(audio))
    upsampled_i[::32]=subband_i
    # synthesis filters are the symmetric of the analysis filters, which
    # ensures linear phase in each sub-band
    Gi_output=signal.lfilter(PQMF32_Gfilters[i,:],1,upsampled_i)
    output_signal=output_signal+Gi_output
    # Screen output
    print('processing sub-band ', i)
    if i==2: # indexing correction
        G3_output=Gi_output

# %%
'''
As revealed by listening sub-band 3, isolated sub-band signals
are very much aliased, because each  PQMF filter is not ideal. 
'''
spectrogram.plot_spectrogram(G3_output,1024,sample_rate,256)
sd.play(G3_output,Fs)

# %%
'''
The PQMF filter bank makes sure aliasing in adjacent bands cancels itself
when sub-bands are added. 
'''
spectrogram.plot_spectrogram(output_signal,1024,Fs,256)
sd.play(output_signal,Fs)

# %%
'''
The power of the reconstruction error is about 85 dB below that of the
signal. Notice the ouput is delayed by 511 samples (since H and G filters
have a delay of 511/2 samples).

*MATLAB function involved:*
 
* |snr = snr(signal,signal_plus_noise, max_shift,showplot)| returns the
signal-to-noise ratio computed from the input signals. |Max_shift| gives
the maximum time-shift (in samples) between |signal| and
|signal_plus_noise|. The actual time-shift (obtained from the maximum of
the cross-correlation between both signals) is taken into account to
estimate the noise. If |showplot| is specified, then the signal,
signal_plus_noise, and error are plotted, and the SNR is printed on the
plot.
'''
error = output_signal[511:] - audio[:-511]

signal_w, signal_Pxx = signal.periodogram(audio[11000:12024], fs=2*np.pi, window=np.hamming(1024), detrend=False, nfft=1024)
error_Pxx_w, error_Pxx = signal.periodogram(error[11000:12024], fs=2*np.pi, window=np.hamming(1024), detrend=False, nfft=1024)

plt.figure(figsize=(10, 6))
plt.plot(signal_w/np.pi*22050, 10 * np.log10(signal_Pxx), label='Signal PSD')
plt.plot(error_Pxx_w/np.pi*22050, 10 * np.log10(error_Pxx), color='r', label='Error PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)
snr_PQMF = snr.snr(audio[:-511], output_signal[511:], 0)
print(f"PQMF Filter Bank SNR: {snr_PQMF:.2f} dB")

# %%
'''
4. Filter banks and Lapped Transforms
If the length of the filters used in M-channel sub-band filters were
exactly M samples, one could easily see the operations performed
simultaneously by the M analysis filters as a linear transform of
successive non-overlapping M sample frames of the input signal.

A 4-sample DFT, for instance, can implement a 4-channel filter bank
whose subband filters are the time-reversed of the lines of the 4x4 DFT
matrix. Applying it to a chirp is straightforward.
'''

Fs=8000
f0 = 0
t1 = 4
f1 = 4000
t  = np.arange(0,4*Fs)/Fs
input_signal=signal.chirp(t,f0,t1,f1)

output_signal = np.zeros(len(input_signal)) # we previously define the size
for i in range(int(len(input_signal)/4)):

    # creating a column vector with 4 samples
    input_frame=input_signal[4*i:4*i+4]
    
    # producing one sample in each downsampled sub-band, i.e. band-pass
    # filtering and downsampling all sub-bands in one operation.
    subbands_sample=np.fft.fft(input_frame)
    
    # producing four samples of the filter bank output, i.e. upsampling, 
    # band-pass filtering all sub-bands, and summing them in one operation. 
    output_frame= np.real(np.fft.ifft(subbands_sample))
    
    # storing the output column vector in the output signal
    output_signal[4*i:4*i+4]=output_frame

sd.play(output_signal,Fs)

# %%

'''
Since the underlying filters have complex coefficients, however, each
sub-band signal is complex. What is more, this type of filter bank is not
very frequency selective, as shown below. The frequency overlap between
adjacent bands is about half the main lobe band-pass (as in the previous
section on PQMF), but the side lobes are very high. This does not make it
a good candidate for sub-band coding. 
'''
tmp = np.array([1,np.exp(-1j*np.pi/2),np.exp(-1j*np.pi),np.exp(-1j*3*np.pi/2)])
DFT_matrix_4x4=np.vander(tmp, increasing=True)

for i in range(4):
    W, H = signal.freqz(DFT_matrix_4x4[i,:], 1, worN=512, whole=True)
    horizontal_axis = np.maximum(20*np.log10(abs(H)), -50) # element wise maximum
    plt.plot(W/np.pi,horizontal_axis)
plt.ylim(-60,20)
plt.xlabel('Normalized frequency (*pi rad/sample)')
plt.ylabel('Magnitude (dB)')

# %%

'''
In general, the length of the impulse responses of the analysis and
synthesis filters used in sub-band coders is higher than the number M of
channels.  The filtering operations, however, can still be
implemented as the multiplication of L-sample frames with LxM or
MxL matrices. 

For example, the 32-channel PQMF filter-bank introduced in the previous
Section, in which the length L of the impulse response of each filter is
512 samples, can be efficiently implemented as follows (which is very much
similar to the implementation of our previous DFT-based filter bank,
with the addition of overlap).
'''
# Build the PQMF H and G filters
# 16 to 537 element list:
elements_16_to_528 = np.arange(16,528,1)
hn = PQMF32_prototype.PQMF32_prototype()
PQMF32_Gfilters = np.zeros((32, 512))

for i in range(32):
    t2 = np.multiply(((2*i+1)*np.pi/(2*32)),elements_16_to_528)
    PQMF32_Gfilters[i,:] = np.multiply(hn, np.cos(t2)) # element wise multiplication

sample_rate, audio_raw = wavfile.read('../audio_samples/violin.wav')
input_signal = audio_raw / 32768

# Block-based sub-band filtering
input_frame=np.zeros(512)
output_signal=np.zeros(np.size(input_signal))

for i in range(int((len(input_signal)-512+32)/32)):
    
    # Overlap input_frames (column vectors)
    input_frame=input_signal[i*32:i*32+512]

    # Analysis filters and downsampling
    # Since PQMF H filters are the time-reversed G filters, we use the G
    # filters matrix to simulate analysis filtering
    subbands_frame_i = np.matmul(PQMF32_Gfilters, input_frame)
    
    # Synthesis filters
    output_frame = np.matmul(np.transpose(PQMF32_Gfilters),subbands_frame_i)

    # Overlap output_frames (with delay of 511 samples)
    output_signal[i*32:i*32+512]= output_signal[i*32:i*32+512]+output_frame

# %%

'''
Obviously we get the same results as before, and the overall SNR is
unchanged.
'''
error=output_signal-input_signal
w, signal_psd = signal.periodogram(input_signal[11000:12024], fs=2*np.pi, window=np.hamming(1024), detrend=False, nfft=1024)
w, error_psd = signal.periodogram(error[11000:12024], fs=2*np.pi, window=np.hamming(1024), detrend=False, nfft=1024)
plt.plot(w/np.pi*22050,10*np.log10(np.maximum(signal_psd,1e-50)), label='Signal PSD')
plt.plot(w/np.pi*22050,10*np.log10(np.maximum(error_psd,1e-50)),color='red',label='Error PSD')
plt.xlim(0,22050)
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()

snr_lapped=snr.snr(input_signal[512:-512],output_signal[512:-512],0)

# %%
'''
5. Perceptual audio coding
The sub-band filtering process developed in the previous Sections
transforms the original stream of samples at sampling frequency Fs into
32 parallel sub-bands sampled at Fs/32. It does not by itself produce
compression.

Quantizing sub-band samples uniformly does not allow much transparency at
low bit rates, as shown in the 4-bits per sub-band sample trial below
(compression factor = 4). We use a mid-thread quantizer here, so as to
encode low level signals to 0.
'''
elements_16_to_528 = np.arange(16,528,1)
# Build the PQMF H and G filters
hn=PQMF32_prototype.PQMF32_prototype()
PQMF32_Gfilters = np.zeros((32, 512))
for i in range(32):
    t2 = np.multiply(((2*i+1)*np.pi/(2*32)),elements_16_to_528)
    PQMF32_Gfilters[i,:] = np.multiply(hn,np.cos(t2))

Fs, audio_raw = wavfile.read('../audio_samples/violin.wav')
input_signal = audio_raw / 32768

# Block-based sub-band analysis filtering
input_frame=np.zeros(512)
output_signal=np.zeros(np.size(input_signal))

n_frames=(len(input_signal)-512+32)/32
subbands = np.zeros((int(n_frames), 32))
quantized_subbands = np.zeros((int(n_frames), 32))
for i in range(int(n_frames)):
     
     # Overlap input_frames (column vectors)
     input_frame=input_signal[i*32:i*32+512]
 
     # Analysis filters and downsampling
     # NB: we put sub-band signals in columns
     subbands[i,:] = np.transpose(np.matmul(PQMF32_Gfilters,input_frame))
 
     # Uniform quantization on 4 bits, using a mid-thread quantizer in
     # [-1,+1] 
     n_bits = 4
     alpha = 2**(n_bits-1)
     quantized_subbands[i,:] = (np.floor(alpha*subbands[i,:]+0.5))/alpha # mid-thread

     # the |uencode| and |udecode| functions provided by MATLAB do not
     # properly implement a mid-thread quantizer. Using them here (by
     # uncommenting the next two lines results in tonal quantization noise.
     # See Appendix 1 at the end of this script. 
     # codes = uencode(subbands(i,:),4,1);
     # quantized_subbands(i,:) = udecode(codes,4);
 
     # Synthesis filters
     output_frame = np.matmul(np.transpose(PQMF32_Gfilters),quantized_subbands[i,:])
 
     # Overlap output_frames (with delay of 511 samples)
     output_signal[i*32:i*32+512]= output_signal[i*32:i*32+512]+output_frame

spectrogram.plot_spectrogram(output_signal,1024,Fs,256)
sd.play(output_signal,Fs)

# %%
'''
The resulting output signal is degraded. It exhibits a strong high
frequency tonal noise. This is typical of sub-band coding, in which
quantization errors are produced at Fs/32, i.e. 1378 Hz.
The SNR falls down to 10.3 dB.

NB: no delay compensation required, as the first sub-band samples
produced by the lapped transform correspond to the 512th original
samples.
'''
error=output_signal-input_signal

w,signal_psd=signal.periodogram(input_signal[11000:12024],fs=2*np.pi, window=np.hamming(1024), detrend=False, nfft=1024)
w,error_psd=signal.periodogram(error[11000:12024],fs=2*np.pi, window=np.hamming(1024), detrend=False, nfft=1024)
plt.plot(w/np.pi*22050,10*np.log10(signal_psd), label='Signal PSD')
plt.plot(w/np.pi*22050,10*np.log10(error_psd),color='red', label='Error PSD')
plt.xlim(0,22050)
plt.legend()
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Magnitude (dB)')

snr_4bits=snr.snr(input_signal[512:-512],output_signal[512:-512],0)

# %%
'''
One can see that the fixed [-1,+1] quantizer range does not adequately
account for the  variation of sub-band signal level across sub-bands, 
as well as in time.  
'''
# [100:200,1] grabs secodn column (python is 0-indexed!)
plt.plot(quantized_subbands[99:200,1],label='Sub-band #2, quantized')
plt.plot(subbands[99:200,1],color='red',linestyle='dashed',label='Sub-band#2, original')
plt.legend()
plt.xlabel('Time (samples at Fs/32)')
plt.ylabel('Amplitude')

# %%
'''
An obvious means of enhancing its quality is therefore to apply a
scale factor to each sub-band quantizer. As in the MPEG-1 Layer-I coder,
we compute a new scale factor every 12 sub-band sample (i.e. every 
32x12 = 384 sample at the original sample rate).
Quantization errors are much reduced (here in sub-band #2).
'''
# Adaptive quantization per blocks of 12 frames
n_frames=np.trunc(n_frames/12)*12 # trunc rounds each element to the nearest integer towards zero
for k in range(0,int(n_frames),12):
    # Computing scale factors in each 12 samples sub-band chunk
    scale_factors=np.max(np.abs(subbands[k:k+12,:]),axis=0)
    
    # Adaptive uniform quantization on 4 bits, using a mid-thread quantizer in
    # [-Max,+Max] 
    for j in range(32): # for each sub-band
 
        n_bits = 4
        alpha = 2**(n_bits-1)/scale_factors[j]
        quantized_subbands[k:k+12,j] = (np.floor(alpha*subbands[k:k+12,j]+0.5))/alpha # mid-thread

plt.plot(quantized_subbands[99:200,1], label='Sub-band #2, quantized')
plt.plot(subbands[99:200,1], color='red',linestyle='dashed',label='Sub-band#2, original')
plt.legend()
plt.xlabel('Time (samples at Fs/32)')
plt.ylabel('Amplitude')

# %%
'''
The resulting signal is of much higher quality.
'''
# Signal synthesis
output_signal=np.zeros(np.size(input_signal))
for i in range(int(n_frames)):

    # Synthesis filters
    output_frame = np.matmul(np.transpose(PQMF32_Gfilters),np.transpose(quantized_subbands[i,:]))

    # Overlap output_frames (with delay of 511 samples)
    output_signal[i*32:i*32+512]= output_signal[i*32:i*32+512]+output_frame

spectrogram.plot_spectrogram(output_signal,1024,Fs,256)
sd.play(output_signal,Fs)

# %%
'''
The overall SNR has increased to 25 dB.
'''
error=output_signal-input_signal
w,signal_psd=signal.periodogram(input_signal[11000:12024],fs=2*np.pi, window=np.hamming(1024), detrend=False, nfft=1024)
w,error_psd=signal.periodogram(error[11000:12024],fs=2*np.pi, window=np.hamming(1024), detrend=False, nfft=1024)
plt.plot(w/np.pi*22050,10*np.log10(signal_psd),label='Signal PSD')
plt.plot(w/np.pi*22050,10*np.log10(error_psd),color='red',label='Error PSD')
plt.xlim(0,22050)
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')

snr_4bits_scaled=snr.snr(input_signal[512:-512],output_signal[512:-512],0)

# %%
'''
The ultimate refinement, which is by far the most effective and results
from years of audio research, consists in accepting more quantization
noise (by allocating less bits, and thereby accepting a higher SNR) in
frequency bands where it will not be heard, and using these extra bits
for more perceptually prominent bands. The required perceptual
information is provided by a psycho-acoustical model.

For any 512-sample frame taken from the input signal, the MPEG-1 Layer-I
psycho-acoustical model computes a global masking threshold, obtained by
first detecting prominent tonal and noise maskers, separately, and
combining their individual thresholds. The maximum of this global
threshold and the absolute auditory threshold is then taken as the final
threshold. 

*MATLAB function involved:*
 
* |function [SMR, min_threshold_subband, masking_threshold] = ...
    MPEG1_psycho_acoustic_model1(frame)|
Computes the masking threshold (in dB) corresponding to psycho-acoustic
model #1 used in MPEG-1 Audio (cf  ISO/CEI  norm 11172-3:1993 (F), pp.
122-128). 
Input |frame| length should be 512 samples. 
|min_threshold_subband| returns the minimun of |masking threshold| in each
of the 32 sub-bands. 
|SMR| returns 27 signal-to-mask ratios (in dB);
|SMR|(28-32) are not used. 
'''
LTq_i = np.array([])
LTq_k = np.array([])
Table_z = np.array([])
Frontieres_i = np.array([])
Frontieres_k = np.array([])
Larg_f = np.array([])

frame=input_signal[11000:11512]
SMR, min_threshold,frame_psd_dBSPL,masking_threshold= mpeg_pam.MPEG1_psycho_acoustic_model1(frame)
# NB: the power levels returned by this function assume that a full-scale 
# signal (in [-1,+1]) corresponds to 96 DB SPL
f_zeros = np.arange(256)
f = f_zeros/512*44100
auditory_threshold_dB = 3.64*((f/1000)**-0.8)- 6.5*np.exp(-0.6*((f/1000)-3.3)**2) + 0.001*((f/1000)**4)
plt.plot(f, frame_psd_dBSPL,label='Signal PSD')
plt.plot(f, min_threshold,'.r',label='Min. threshold per sub-band')
plt.plot(f, auditory_threshold_dB, '-.k', label='Absolute threshold')
plt.ylim(-20, 100)
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')

# %%
'''
Signal-to-Mask Ratios (SMR) are computed for each band, in a very
conservative way (as the ratio of the maximum of the signal PSD to the
minimum of the masking threshold in each band).  
Bit allocation is performed by an iterative algorithm which gives
priority to sub-bands with higher SMR. 
The resulting SNR in each sub-band should be greater or equal to
the SMR, so as to push the noise level below the masking threshold.

*MATLAB function involved:*
 
* |function [N_bits,SNR] = MPEG1_bit_allocation(SMR, bit_rate)|
Implements a simplified bit allocation greedy algorithm.
|SMR| is the signal-to-mask ratios in each sub-band,
as defined by the MPEG1 psycho-acoustic model. 
|bit_rate| is in kbits/s.
|N_bits| is the number is bits in each sub_band.
|SNR| is the maximum SNR in each sub-band after quantization, i.e. the
SNR assuming each sub-band contains a full-range sinusoid.
NB: N_bits and SNR are set to zero for sub-bands 28 to 32.
'''
# Allocating bits for a target bit rate of 192 kbits/s (compression
# ratio = 4)
N_bits, SNR = mpeg_ba.MPEG1_bit_allocation(SMR, 192000)

x_axis=np.arange(33)
x_axis=(x_axis/32)*22050
plt.stairs(SMR,x_axis ,label='SMR')
plt.stairs(SNR,x_axis,label='SNR')
plt.xlim(0,22050)
plt.ylim(-20,100)
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')

# %%
'''
Let us test this on the complete signal, adding perceptual bit allocation
to adaptive uniform quantization. 
Notice we simplify the quantization
scheme here, compared to MPEG-1, by considering quantization with any
number of bits in [0,16].  
'''
# Adaptive quantization per blocks of 12 frames
n_frames=np.trunc(n_frames/12)*12 
for k in range(0,int(n_frames),12):
    # Computing scale factors in each 12 samples sub-band chunk
    scale_factors=np.max(np.abs(subbands[k:k+12,:]),axis=0)
    
    # Computing SMRs
    frame=input_signal[175+k*32:175+k*32+512]
    # NB: the input frame for the psycho-acoustic model is delayed by 176
    # samples, as it should be centered in the middle of the local block of
    # 12 sub-band samples, and the first sub-band sample corresponds to
    # original sample 256 (actually, to sample 512, but the analysis filter
    # introduces a delay of 256 samples). So the center of the first frame should
    # be on original sample 256+11*32/2=432, and the beginning of this frame
    # falls on sample 11*32/2=176.
    SMR, min_threshold,frame_psd_dBSPL,masking_threshold = mpeg_pam.MPEG1_psycho_acoustic_model1(frame)

    # Allocating bits for a target bit rate of 192 kbits/s (compression ratio = 4)
    N_bits, SNR = mpeg_ba.MPEG1_bit_allocation(SMR, 192000)

    # Adaptive perceptual uniform quantization, using a mid-thread 
    # quantizer in [-Max,+Max] 
    for j in range(32):
        if N_bits[j]!=0:
            alpha = 2**(N_bits[j]-1)/scale_factors[j]
            quantized_subbands[k:k+11,j] = np.floor(alpha*subbands[k:k+11,j]+0.5)/alpha # mid-thread

            # codes = uencode(subbands[k:k+11,j],N_bits[j],scale_factors[j],'signed')
            # quantized_subbands[k:k+11,j]=udecode(codes,N_bits[j],scale_factors[j])
        else:
            quantized_subbands[k:k+11,j] = 0

    # Screen output
    print(f'processing frame {int(k+1):3d}/{int(n_frames):3d}')

# Signal synthesis
output_signal=np.zeros_like(input_signal)
for i in range(int(n_frames)):
    # Synthesis filters
    output_frame = np.dot(np.transpose(PQMF32_Gfilters), quantized_subbands[i, :])
    # Overlap output_frames (with delay of 511 samples)
    output_signal[i*32:i*32+512]= output_signal[i*32:i*32+512]+output_frame

spectrogram.plot_spectrogram(output_signal,1024,Fs,256)
sd.play(output_signal,Fs)

# %%
'''
Quantization noise is now very small in some prominent sub-bands, like
sub-band #2. 
'''
plt.plot(subbands[99:200,1],'--r', label='Sub-band #2, original')
plt.plot(quantized_subbands[99:200,1], label='Sub-band#20, quantized')
plt.legend()
plt.xlabel('Time (samples at Fs/32)')
plt.ylabel('Amplitude')

# %%
'''
It is also more important in other sub-bands, like sub-band #20. 
'''
plt.plot(subbands[99:200,19],'--r', label='Sub-band #20, original')
plt.plot(quantized_subbands[99:200,19],label='Sub-band#20, quantized') 
plt.legend()
plt.xlabel('Time (samples at Fs/32)')
plt.ylabel('Amplitude')

# %%
'''
The overall SNR has increased to 36.2 dB.
The perceptual SNR is actually much higher, since most of the noise
cannot be heard. 
'''
error=np.subtract(output_signal,input_signal)
for i in range(len(error)):
    error[i]=np.round(error[i],4)

w, signal_psd = signal.periodogram(input_signal[11000:12024],
        fs=2*np.pi, window=np.hamming(1024), detrend=False, nfft=1024)
W, error_psd = signal.periodogram(error[11000:12024],
        fs=2*np.pi, window=np.hamming(1024), detrend=False, nfft=1024)
plt.plot(w/np.pi*22050,10*np.log10(signal_psd),label='Signal PSD')
plt.plot(w/np.pi*22050,10*np.log10(error_psd),'r',label='Error PSD')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')

snr_scaled_perceptual=snr.snr(input_signal[512:-512],output_signal[512:-512],0)

# %%
'''
The price to pay is that the scale factors and the number of bits per
sub-band must now be stored, every 12 frames (i.e. every 12*32 sub-band
samples). 
In the MPEG-1 norm, scale factors expressed in dB and quantized
on 6 bits each. This comes from the fact that the ear perceives loudness
as the log of the energy, and has about 96 dB of hearing dynamics with a
sensitivity threshold of about 1 dB. With 6 bits, the quantization step
is 96/64 dB and the error lies in [-96/62/2, +96/64/2], which is below
the 1 dB threshold.
Assuming 4 bits are required for this information in each of the
27 first sub-bands (sub-bands 28 to 32 are ignored my MPEG1), this leads
to 27*10=270 bits every 12 frames. 
(In practice, MPEG-1 does not allow all integer values in [1,16] for the
number of bits in each band, which makes it possible to quantize it to
less then 4 bits. The total number of bits used for bit allocation is
then reduced to 88.) 
It is easy to obtain the number of bits used for the last block of 12
frames in our audio test, and the related bit-rate.
'''
bits_per_block=np.sum(N_bits)*12+270
bit_rate=bits_per_block*44100/384
# %%
'''
Appendix 1: Uniform quantizers
In Section 5 we have used a mid-thread quantizer. It was tempting to use
the |uencode| and |udecode| fucntions provided by MATLAB, but a quick
examination of the quantization charactertic of these functions shows
they do not implement a real mid-thread quantizer.
'''
def uencode(u, n, v):
    # Clipping:
    # stays same if -v < element < v
    # v if element was greater than v
    # -v if element was smaller than -v
    u = np.clip(u, -v, v)
    L = 2**n # number of quantization levels
    quantized = np.floor((u+v)*L/(2*v)) # shift,scale,normalize
    quantized = np.clip(quantized,0,L-1)
    return quantized

def udecode(y, n, v):
    L = 2**n
    # opposite operations of encoding
    x = (y)*(2*v)/L-v
    return x

x = np.arange(-1,1,0.01)
n_bits=3
codes = uencode(x,n_bits,1)
y1 = udecode(codes,n_bits,1)
alpha = 2**(n_bits-1)
y2 = (np.floor(alpha*x)+0.5)/alpha # mid-rise
y3 = (np.floor(alpha*x+0.5))/alpha # mid-thread

plt.plot(x,y1,'b',label='uen(de)code')
plt.plot(x,y2,'r',label='mid-rise')
plt.plot(x,y3,'g',label='mid-thread')
plt.plot([0, 0], [-1, 1],color='g')
plt.plot([-1, 1], [0, 0], color='purple')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.legend()

# %%