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
upsampled = np.zeros(2*len(downsampled))
upsampled[::2]=2*downsampled
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
plot_spectrogram(G0_output[1000:],1024,Fs,512)

# A audio player will be implemented here.

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
plot_spectrogram(G0_output[2000:],1024,Fs,256)

# An audio player will be implemented here.

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

plot_spectrogram(G1_output[2000:],1024,Fs,256)
# An audio player will be implemented here.

# %%

'''
We now examine the output of the 2-channel sub-band filter bank, by
adding  the two signals obtained above.
'''

synt_signal = G0_output + G1_output
plot_spectrogram(synt_signal[2000:],1024,Fs,256)
# An audio player will be implemented here.

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
plt.show()

# An audio player will be implemented here.

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
print(H0_QMF_reverse)
print(H0_QMF)
H0_QMF_extended=H0_QMF + H0_QMF_reverse
print(H0_QMF_extended)
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
plot_spectrogram(G0_output,1024,Fs,256)
# An audio player will be implemented here.

# %%
# HF band

H1_output=signal.lfilter(H1_QMF,1,input_signal)
subband_1=H1_output[::2]
upsampled = np.zeros(2*len(subband_0))
upsampled[::2]=2*subband_1
G1_QMF=-H1_QMF
G1_output=signal.lfilter(G1_QMF,1,upsampled)

plot_spectrogram(G1_output,1024,Fs,256)
# An audio player will be implemented here.

# %%
'''
Perfect reconstruction is now achieved, because the QMF analysis and
synthesis filters are such that the aliasing in each band sum up to zero
(more precisely, close to 0). 
'''
synt_signal=G0_output+G1_output
plot_spectrogram(synt_signal,1024,Fs,256)
# An audio player will be implemented here.

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

def PQMF32_prototype():
    '''
    hn = PQMF32_prototype returns in hn the impulse response of the prototype
    low-pass symmetric filter of length 512 for building a 32-channel PQMF
    filter bank. This filter is used in the MPEG-1 Layer-1 coder. Its
    normalized bandpass is 1/64 Hz and satisfies the PR condition.   
    '''
    # Second half of the filter
    temp = [0.202407,  0.202283,  0.201916,  0.201307,
            0.200452,  0.199359,  0.198029,  0.196465,
            0.194668,  0.192648,  0.190409,  0.187952,
            0.185290,  0.182422,  0.179361,  0.176113,
            0.172685,  0.169084,  0.165321,  0.161407,
            0.157347,  0.153153,  0.148837,  0.144402,
            0.139868,  0.135239,  0.130527,  0.125745,
            0.120900,  0.116004,  0.111068,  0.106105,
            0.101123,  0.096135,  0.091148,  0.086174,
            0.081224,  0.076307,  0.071432,  0.066610,
            0.061849,  0.057155,  0.052540,  0.048011,
            0.043576,  0.039242,  0.035012,  0.030899,
            0.026907,  0.023036,  0.019297,  0.015693,
            0.012227,  0.008901,  0.005724,  0.002692,
            -0.000189, -0.002919, -0.005495, -0.007917,
            -0.010185, -0.012303, -0.014264, -0.016074,
            -0.017733, -0.019243, -0.020608, -0.021827,
            -0.022906, -0.023845, -0.024652, -0.025326,
            -0.025873, -0.026300, -0.026604, -0.026799,
            -0.026882, -0.026863, -0.026747, -0.026537,
            -0.026238, -0.025855, -0.025399, -0.024867,
            -0.024271, -0.023616, -0.022904, -0.022143,
            -0.021336, -0.020492, -0.019613, -0.018706,
            -0.017773, -0.016824, -0.015858, -0.014882,
            -0.013900, -0.012915, -0.011936, -0.010960,
            -0.009994, -0.009039, -0.008103, -0.007183,
            -0.006285, -0.005411, -0.004564, -0.003744,
            -0.002954, -0.002196, -0.001470, -0.000777,
            -0.000121,  0.000499,  0.001084,  0.001632,
            0.002142, 0.002616,  0.003051,  0.003453,
            0.003814,  0.004141,  0.004435,  0.004691,
            0.004915,  0.005106,  0.005265,  0.005395,
            0.005495,  0.005565,  0.005611,  0.005629,
            0.005624,  0.005597,  0.005549,  0.005481,
            0.005397,  0.005292,  0.005176,  0.005044,
            0.004901,  0.004745,  0.004580,  0.004408,
            0.004227,  0.004041,  0.003852,  0.003658,
            0.003461,  0.003264,  0.003067,  0.002870,
            0.002673,  0.002479,  0.002287,  0.002101,
            0.001918,  0.001740,  0.001567,  0.001400,
            0.001238,  0.001082,  0.000936,  0.000793,
            0.000658,  0.000531,  0.000413,  0.000299,
            0.000194,  0.000097,  0.000005, -0.000078,
            -0.000154, -0.000224, -0.000286, -0.000343,
            -0.000394, -0.000440, -0.000477, -0.000510,
            -0.000539, -0.000561, -0.000580, -0.000596,
            -0.000604, -0.000612, -0.000615, -0.000615,
            -0.000612, -0.000607, -0.000599, -0.000588,
            -0.000575, -0.000561, -0.000545, -0.000529,
            -0.000513, -0.000494, -0.000475, -0.000456,
            -0.000434, -0.000415, -0.000397, -0.000375,
            -0.000356, -0.000337, -0.000316, -0.000299,
            -0.000281, -0.000262, -0.000245, -0.000229,
            -0.000213, -0.000197, -0.000183, -0.000170,
            -0.000156, -0.000143, -0.000132, -0.000121,
            -0.000111, -0.000103, -0.000094, -0.000084,
            -0.000078, -0.000070, -0.000065, -0.000057,
            -0.000051, -0.000046, -0.000043, -0.000038,
            -0.000035, -0.000030, -0.000027, -0.000024,
            -0.000022, -0.000019, -0.000019, -0.000016,
            -0.000013, -0.000013, -0.000011, -0.000011,
            -0.000008, -0.000008, -0.000005, -0.000005,
            -0.000005, -0.000005, -0.000003, -0.000003,
            -0.000003, -0.000003, -0.000003, -0.000003]

    # Full symmetric filter response
    a = [0]
    temp_reversed = temp[::-1]
    temp_reversed.pop()
    hn = a+temp_reversed+temp
    print(hn)
    return hn


# Load the prototype lowpass filter
hn = PQMF32_prototype()

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
from scipy.io import wavfile
sample_rate, audio_raw = wavfile.read('../audio_samples/violin.wav')
audio = audio_raw / 32768
total_duration = len(audio)
plot_spectrogram(audio,1024,Fs,256);
# An audio player will be implemented here.

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
plot_spectrogram(G3_output,1024,sample_rate,256)
# An audio player will be implemented here.

# %%
'''
The PQMF filter bank makes sure aliasing in adjacent bands cancels itself
when sub-bands are added. 
'''
plot_spectrogram(output_signal,1024,Fs,256)
# An audio player will be implemented here.

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
def snr(sig,signal_plus_noise,max_shift,showplot):
    '''
    [snr_value, shift] = snr(signal,signal_plus_noise, max_shift,showplot) returns the
    signal-to-noise ratio computed from the input signals. |Max_shift| gives
    the maximum time-shift (in samples) between signal and signal_plus_noise.
    The actual time-shift (obtained from the maximum of the cross-correlation
    and returned as |shift|) is taken into account to estimate the noise. If
    signal are of different length, the shortest length is used.
    If |showplot| is specified, then the signal, signal_plus_noise, and error 
    are plotted, and the SNR is printed on the plot.
    
    T DUTOIT, 13:49 12/03/2007
    '''
    sig_length=np.min(len(sig),len(signal_plus_noise))
    sig_length=np.floor(sig_length/2)*2 # make it even
    sig=sig[1:sig_length]
    signal_plus_noise=signal_plus_noise[1:sig_length]

    len=np.min(np.max(10*max_shift,1000), sig_length-1)
    half_len=np.floor(len/2)
    center=np.floor(sig_length/2)
    cross_correlation = signal.correlate(signal_plus_noise[center-half_len:center+half_len-1],sig[center-half_len:center+half_len-1],max_shift,'biased')
    [max_value,max_lag] = np.max(cross_correlation[max_shift+1:2*max_shift+1])
    shift=max_lag-1

    tmp=signal_plus_noise[1+shift:len(signal_plus_noise)]
    noise=tmp-sig[1:len(tmp)]

    # Uncomment this to see the signals on which the SNR is computed
    # plot(signal(1:length(tmp))); hold on;plot(tmp,'r'); plot(noise,'g'); hold off;
    '''
    plt.plot(sig[1:len(tmp)])
    plt.plot(tmp, 'red')
    plt.plot(noise, 'green')
    '''

    var_signal_dB=10*np.log10(np.var(sig))
    var_noise_dB=10*np.log10(np.var(noise))
    snr_value=var_signal_dB-var_noise_dB

    # Alternative to nargin will be implemented
    '''
    if nargin==4:
        plt.plot(sig[center-half_len:center+half_len-1], label='signal')
        plt.plot(tmp[center-half_len:center+half_len-1],'r',label='signal+noise')
        plt.plot(noise[center-half_len:center+half_len-1],'g', label='noise')
        plt.legend()
        plt.title('SNR = ', num2str(snr_value))
    '''
    return snr_value

error = output_signal[511:]-audio[:-511]

signal_w, signal_Pxx = signal.periodogram(audio[11000:12024],fs=sample_rate,window=np.hamming(1024),detrend=False,nfft=1024)
error_Pxx_w, error_Pxx = signal.periodogram(error[11000:12024],fs=sample_rate,window=np.hamming(1024),detrend=False,nfft=1024)

plt.plot(signal_w/np.pi*22050,10*np.log10(signal_Pxx), label='Signal PSD')
plt.plot(error_Pxx_w/np.pi*22050,10*np.log10(error_Pxx),'r','linewidth',2, label='Error PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.xlim(0, 22050)
plt.grid(True)
plt.show()
snr_PQMF=snr(audio[:-511],output_signal[511:],0)

# %%