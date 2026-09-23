# %%
'''
Chapter 7 - How could music contain hidden information
This is a companion file to the book "Applied Signal Processing", 
by T.Dutoit and F. Marques, Springer 2008.

Python translation by Ediz Aldogan.

It is supposed to be run cell-by-cell (the "# %%" markers define the cells,
which are understood by VS Code and Jupyter).
'''
# %%
'''
In the following script, we will develop watermarking systems,
whose design is based on classical communication systems using spread
spectrum modulation. We first examine the implementation of a
watermarking system and give a brief overview of its performance in the
simple and theoretical configuration where the audio signal is a white
Gaussian noise (Section 1). Then, we show how this system can be adapted
to account for the audio signal specificities, while still satisfying the
major properties and requirements of watermarking applications (namely
inaudibility, correct detection and robustness of the watermark): we
extend the initial system by focussing on the correct detection of the
watermark (Section 2), on the inaudibility constraint (Section 3), and on
the robustness of the system to MP3 compression (Section 4). 

Copyright C. Baras, N. Moreau, T. Dutoit (2007)
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.linalg import toeplitz

import visual_tools as vt
from tools import (soundsc, xcorr,
                          message_to_bits, bits_to_message)
from psychoacoustical_model import psychoacoustical_model
from shaping_filter_design import shaping_filter_design
from mp3_codec import mp3_codec

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.max_open_warning'] = 0   # many figures are created below

# %%
'''
1. Audio watermarking, seen as a digital communication problem
The following digital communication system is the direct implementation
of the theoretical results detailed in Section 1 of the book chapter. The
watermark message is embedded in the audio signal at the binary rate
|R|=100 bps. To establish the analogy between watermarking system and
communication channel, the audio signal is viewed as the channel noise.
In this section, it will therefore be modeled as a white Gaussian noise
with zero mean and variance |sigma2_x|. |sigma2_x| is computed so that
the Signal-to-Noise Ratio (SNR), i.e. the ratio between the watermark and
the audio signal powers, equals -20 dB.
'''
# %%
'''
1.1. Emitter/Embedder
Let us first generate a watermark message, and its corresponding bit
sequence.
'''

message = ('Audio watermarking: this message will be embedded in an audio '
           'signal, using spread spectrum modulation, and later retrieved '
           'from the modulated signal.')
print(message)
bits = message_to_bits(message)
symbols = bits*2 - 1                     # a (1050,) array of -1's and +1's
N_bits = len(bits)
print('N_bits =', N_bits)

# %%
'''
We then generate the audio signal, the spread spectrum sequence, and the
watermarked signal. The latter is obtained by first deriving a symbol
sequence (-1's and +1's) from the input bit sequence (0's and +1's), and
then by modulating the spread spectrum signal by the symbol sequence. This
is achieved by concatenating spread spectrum waveforms weighted by the
symbol values. 
'''
# Random sequences generators initialization
np.random.seed(12345)

# Sampling frequency |Fs|, binary rate |R|, samples per bit |N_samples| and |SNR|
Fs = 44100
R = 100
N_samples = int(round(Fs/R))
SNR_dB = -20

# Spread spectrum waveform: a random sequence of -1's and +1's
spread_waveform = 2*np.round(np.random.rand(N_samples))-1

# Audio signal: a Gaussian white noise with fixed variance
sigma2_x = 10**(-SNR_dB/10)
audio_signal = np.sqrt(sigma2_x)*np.random.randn(N_bits*N_samples)

# Modulated signal and watermark signal
modulated_signal = np.zeros(N_bits*N_samples)
for m in range(N_bits):
    modulated_signal[m*N_samples:m*N_samples+N_samples] = \
        symbols[m]*spread_waveform

watermark_signal = modulated_signal  # no gain applied

# Plotting the baseband signal (i.e., the emitted signal corresponding
# to the input bit sequence) and the watermark signal

emitted_signal = np.repeat(symbols, N_samples)

axe_ech = np.arange(4*N_samples, 7*N_samples+2)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(axe_ech/Fs, emitted_signal[axe_ech-1])
plt.axis([axe_ech[0]/Fs, axe_ech[-1]/Fs, -1.1, 1.1])
plt.xlabel('Time(s)')

plt.subplot(2, 1, 2)
plt.plot(axe_ech/Fs, watermark_signal[axe_ech-1])
plt.axis([axe_ech[0]/Fs, axe_ech[-1]/Fs, -1.1, 1.1])
plt.xlabel('Time(s)')

# %%
''' 
As expected, the power spectral density (PSD) of the spread spectrum
sequence (i.e., the watermark signal) is much flatter than that of the
baseband signal (i.e., the emitted signal) corresponding to the input
bit sequence, while their power is identical: 0dB.
'''
power_baseband_dB = 10*np.log10(np.var(emitted_signal))
power_spread_spectrum_dB = 10*np.log10(np.var(watermark_signal))
print('power_baseband_dB = %.2f' % power_baseband_dB)
print('power_spread_spectrum_dB = %.2f' % power_spread_spectrum_dB)

# NB: we claim Fs=2 so as to make pwelch return |FFT²/N|, in which the
# variance of a white noise is directly readable. See the appendix in
# ASP_audio_cd.py for more details.

plt.figure()
plt.subplot(2, 1, 1)
vt.plot_pwelch(emitted_signal, fs=2, new_figure=False)
plt.xlabel('Normalized Frequency : f/(Fs/2)')
plt.subplot(2, 1, 2)
vt.plot_pwelch(watermark_signal, fs=2, new_figure=False)
plt.xlabel('Normalized Frequency : f/(Fs/2)')
plt.tight_layout()

# %%
'''
The embedding process finally consists in adding the result to the audio
signal, yielding the audio watermarked signal.
'''
watermarked_signal = watermark_signal + audio_signal

# Plotting the first 100 samples of the watermark, audio, and watermarked
# signals shows that the watermarked signal is only slightly different from
# the audio signal, given the SNR we have imposed.
t_100 = np.arange(1, 101)/Fs

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t_100, watermark_signal[:100], 'k-')
plt.ylim([-1.1, 1.1])

plt.subplot(3, 1, 2)
plt.plot(t_100, audio_signal[:100])

plt.subplot(3, 1, 3)
plt.plot(t_100, watermarked_signal[:100], 'r')
plt.xlabel('Time(s)')

# %%
'''
1.2. Receiver
The watermark receiver is a correlation demodulator. It computes the
normalized scalar product |alpha| between frames of the received signal
and the spread spectrum waveform, and decides on the received bits, based
on the sign of |alpha|. One can see on the plot (for bits 81 to 130) that
the audio signal sometimes imposes a wrong sign to |alpha|, leading to
erroneous bit detection.
'''
alpha = np.zeros(N_bits)
received_bits = np.zeros(N_bits, dtype=int)
for m in range(N_bits):
    alpha[m] = np.dot(watermarked_signal[m*N_samples:m*N_samples+N_samples],
                      spread_waveform)/N_samples
    if alpha[m] <= 0:
        received_bits[m] = 0
    else:
        received_bits[m] = 1

# Plotting the input bits, alpha, and the received bits.
rng = np.arange(80, 130)

plt.figure()
plt.subplot(3, 1, 1)
plt.step(rng+1, bits[rng], where='post')
plt.ylim([-0.1, 1.1])
plt.subplot(3, 1, 2)
plt.step(rng+1, alpha[rng], where='post')
plt.ylim([-3, 3])
plt.subplot(3, 1, 3)
plt.step(rng+1, received_bits[rng], where='post')
plt.ylim([-0.1, 1.1])
plt.xlabel('index')

# %%
'''
The performance of the receiver can be estimated by computing the Bit
Error Rate (BER) and by decoding the received message. As can be observed
on the message, a bit error rate of .02 is disastrous in terms of message
understandability.
'''
number_of_erroneous_bits = np.sum(bits != received_bits)
total_number_of_bits = N_bits
BER = number_of_erroneous_bits/N_bits
print('number_of_erroneous_bits =', number_of_erroneous_bits)
print('total_number_of_bits =', total_number_of_bits)
print('BER =', BER)

received_message = bits_to_message(received_bits)
print('received_message =', received_message)

# %%
'''
1.3. System performance overview
A good overview of the system performance can be drawn from the
statistical observation of |alpha| values, through a histogram plot. As
expected, a bimodal distribution is found, with non zero overlap.
Notice that the histogram only gives a rough idea of the underlying 
distribution, since the number of emitted bits is small. 
'''
plt.figure()
plt.hist(alpha, 50)
plt.xlabel('alpha')

# %%
'''
In our particular configuration, the spread waveform is a realization of
a random sequence with values in {+1,-1}, and the audio-noise vectors
are modeled by |N_samples| independent Gaussian random variables with
zero mean and variance |sigma2_x|. In this case, it can be shown that the
Probability Density Function (PDF) of |alpha| is the average of two
normal distributions with mean |a*g| (|a|=+1 or -1) and variance
|sigma2_x/N_samples|. 
Plotting the theoretical normal distribution shows a better view of the
overlap area observed on the histogram. In particular, the area of the
overlap between the two Gaussian modes corresponds to the theoretical BER
(the exact expression of this BER is given in the main text). 
'''
alpha_range = np.arange(-3, 3.0001, .1)
sigma2_alpha = sigma2_x/N_samples
gauss0 = 1/2*np.exp(-((alpha_range-1)**2)/(2*sigma2_alpha)) \
    / np.sqrt(2*np.pi*sigma2_alpha)
gauss1 = 1/2*np.exp(-((alpha_range+1)**2)/(2*sigma2_alpha)) \
    / np.sqrt(2*np.pi*sigma2_alpha)
theoretical_PDF = gauss0+gauss1

plt.figure()
plt.plot(alpha_range, gauss0)
plt.plot(alpha_range, gauss1)

ind0 = np.where(alpha_range <= 0)[0]
ind1 = np.where(alpha_range >= 0)[0]
plt.fill_between(alpha_range[ind1], gauss1[ind1], color=[0.39, 0.47, 0.64])
plt.fill_between(alpha_range[ind0], gauss0[ind0], color=[0.39, 0.47, 0.64])
plt.xlabel('alpha')

theoretical_BER1 = 2*np.sum(gauss1[ind1])*(alpha_range[1]-alpha_range[0])
print('theoretical_BER1 =', theoretical_BER1)

# %%
'''
2. Informed watermarking with error-free detection
The previous system did not take the audio signal specificities into
account (since audio was modeled as white Gaussian noise). We now 
examine how to modify the system design to reach one major requirement
for a watermarking application: that of detecting the watermark message
without error. For this purpose, the watermark gain (also called
embedding strength) has to be adjusted to the local audio variations.
This is referred to as informed watermarking.
'''
# Preparing the data, as in Section 1.
bits = message_to_bits(message)
symbols = bits*2 - 1
N_bits = len(bits)
np.random.seed(12345)
Fs = 44100
R = 100
N_samples = int(round(Fs/R))
SNR_dB = -20
spread_waveform = 2*np.round(np.random.rand(N_samples))-1   # in {-1,1}
modulated_signal = np.zeros(N_bits*N_samples)
for m in range(N_bits):
    modulated_signal[m*N_samples:m*N_samples+N_samples] = \
        symbols[m]*spread_waveform

# From now on, the audio signal will be a violin signal sampled at
# |Fs|=44.100 Hz, of which only the first |N_bits*N_samples| samples will
# be watermarked. This signal is normalized in [-1,+1].
Fs, audio_raw = wavfile.read('../../audio_samples/violin.wav')
audio_signal = audio_raw[:N_bits*N_samples] / 32768

soundsc(audio_signal, Fs)

# %%
'''
2.1. Informed emitter
To reach a zero BER, the watermark gain is adapted so that the
correlation between the watermarked audio signal and the spread sequence 
is at least equal to some security margin |Delta_g| (if |am|=+1)
and |-Delta_g| (if |am|=-1). |Delta_g| sets up a robustness margin
against additive perturbation (such as the additive noise introduced by 
MPEG compression of the watermarked audio signal).
In the following implementation, |Delta_g| is empirically set to 0.005. 
'''
Delta_g = 0.005
gain = np.zeros(N_bits)
watermark_signal = np.zeros(N_bits*N_samples)

for m in range(N_bits):
    beta = np.dot(audio_signal[m*N_samples:m*N_samples+N_samples],
                  spread_waveform)/N_samples
    if symbols[m] == 1:
        if beta >= Delta_g:
            gain[m] = 0
        else:
            gain[m] = Delta_g - beta
    else:                                    # if symbols[m] == -1
        if beta <= -Delta_g:
            gain[m] = 0
        else:
            gain[m] = Delta_g + beta
    watermark_signal[m*N_samples:m*N_samples+N_samples] = \
        gain[m]*modulated_signal[m*N_samples:m*N_samples+N_samples]

watermarked_signal = watermark_signal + audio_signal  # Watermarked audio

# %%
'''
Plotting the gain for one second of signal, together with the resulting
watermark signal and the audio signal, shows that the encoder has to
make important adjustments as a function of the audio signal.
'''
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(np.arange(100)*N_samples/Fs, gain[:100])
plt.ylabel('gain')
plt.subplot(2, 1, 2)
plt.plot(np.arange(Fs)/Fs, watermark_signal[:Fs])
plt.xlabel('time (s)')

# %%
'''
The watermark, though, is small compared to the audio signal.
'''
plt.figure()
plt.plot(np.arange(Fs)/Fs, audio_signal[:Fs], label='audio signal')
plt.plot(np.arange(Fs)/Fs, watermark_signal[:Fs], 'r-',
         label='watermark signal')
plt.xlabel('time (s)')
plt.legend()

# %%
'''
Plotting again a few samples of the watermark, audio, and watermarked
signals shows that the watermarked signal sometimes differs
significantly from the audio signal. This is confirmed by a listening
test: the watermark is audible.
'''
rng = np.arange(34738, 34938)

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(rng/Fs, watermark_signal[rng], 'k-')
plt.xlim([rng[0]/Fs, rng[-1]/Fs])
plt.subplot(3, 1, 2)
plt.plot(rng/Fs, audio_signal[rng])
plt.xlim([rng[0]/Fs, rng[-1]/Fs])
plt.subplot(3, 1, 3)
plt.plot(rng/Fs, watermarked_signal[rng], 'r')
plt.xlim([rng[0]/Fs, rng[-1]/Fs])
plt.xlabel('Time(s)')

soundsc(watermarked_signal, Fs)

# %%
'''
2.2. Receiver
Using the same correlation demodulator as in Section 1, we conclude that the
transmission is now effectively error-free, as confirmed by the resulting
BER.
'''
alpha = np.zeros(N_bits)
received_bits = np.zeros(N_bits, dtype=int)
for m in range(N_bits):
    alpha[m] = np.dot(watermarked_signal[m*N_samples:m*N_samples+N_samples],
                      spread_waveform)/N_samples
    received_bits[m] = 0 if alpha[m] <= 0 else 1

# Plotting the input bits, the value of alpha, and the received bits.
rng = np.arange(80, 130)

plt.figure()
plt.subplot(3, 1, 1)
plt.step(rng+1, bits[rng], where='post')
plt.ylim([-0.1, 1.1])
plt.subplot(3, 1, 2)
plt.step(rng+1, alpha[rng], where='post')
plt.ylim([-.01, .01])
plt.subplot(3, 1, 3)
plt.step(rng+1, received_bits[rng], where='post')
plt.ylim([-0.1, 1.1])
plt.xlabel('frame index')

number_of_erroneous_bits = np.sum(bits != received_bits)
total_number_of_bits = N_bits
BER = number_of_erroneous_bits/total_number_of_bits
print('number_of_erroneous_bits =', number_of_erroneous_bits)
print('BER =', BER)
print('received_message =', bits_to_message(received_bits))

# %%
'''
3. Informed watermarking made inaudible
The informed watermarking system exposed in Section 2 proves that prior
knowledge of the audio signal can be efficiently used in the watermarking
process: error-free transmission is reached thanks to an adaptive
embedding gain. However, the resulting watermark is audible, since no
perceptual condition is imposed on the embedding gain. In this Section,
we examine how psychoacoustics can be put to profit to ensure the
inaudibility constraint.
A psychoacoustic model is used, which provides a signal-dependent masking
threshold used as an upper bound for the PSD of the watermark. The
watermark gain is therefore replaced by an all-pole perceptual shaping
filter and the reception process is composed of a zero-forcing equalizer,
followed by a linear-phase Wiener filter.
'''
# Preparing the data, as in Section 2.
bits = message_to_bits(message)
symbols = bits*2 - 1
N_bits = len(bits)
np.random.seed(12345)
Fs = 44100
R = 100
N_samples = int(round(Fs/R))
SNR_dB = -20
spread_waveform = 2*np.round(np.random.rand(N_samples))-1   # in {-1,1}
modulated_signal = np.zeros(N_bits*N_samples)
for m in range(N_bits):
    modulated_signal[m*N_samples:m*N_samples+N_samples] = \
        symbols[m]*spread_waveform
Fs, audio_raw = wavfile.read('../../audio_samples/violin.wav')
audio_signal = audio_raw[:N_bits*N_samples] / 32768

soundsc(audio_signal, Fs)

# %%
'''
3.1. Emitter, based on perceptual shaping filtering 
The perceptual shaping filter is an auto-regressive filter with 50
coefficients |ai| and gain |b0|. It is designed so that the PSD of the
watermark (obtained by filtering the |modulated_signal|) equals the
|masking_threshold|. The coefficients of the filter are obtained as in
Chapter 1, via the Levinson algorithm. Both the masking threshold and the
shaping filter have to be updated each time the (statistical) properties
of the |audio_signal| change (here every 512 samples).   

Let us first compute the masking threshold and apply the associated shaping
filter to one audio frame (the 11-th frame, for instance).

Functions involved:

* |masking_threshold = psychoacoustical_model( audio_signal )| returns
the masking threshold deduced from a psychoacoustical analysis of the audio
vector. This implementation is derived from the psycho-acoustic model #1
used in MPEG-1 Audio (see ISO/CEI norm 11172-3:1993 (F), pp. 122-128, or
the MPEG1_psycho_acoustic_model1 function from Chapter 3). It is based on
the same principles as those used in the MPEG model, but it is further
adapted here so as to make it robust to additive noise (which is a specific
constraint of watermarking and is not found in MPEG).

* |b0, ai = shaping_filter_design(desired_frequency_response_dB, N_coef)| 
computes the coefficients of an auto-regressive filter from the modulus of
its desired frequency response (in dB) and the order |N_coef|.
'''
N_coef = 50
N_samples_PAM = 512
PAM_frame = np.arange(10*N_samples_PAM, 10*N_samples_PAM+N_samples_PAM)

# Showing the audio signal frame
plt.figure()
plt.plot(PAM_frame/Fs, audio_signal[PAM_frame])
plt.xlabel('Time (s)')

# %%
'''
Comparing the periodogram of the audio signal, the masking threshold and
the frequency response of the filter shows that the filter closely
matches the masking threshold. 
Even the absolute amplitude level of the filter is the same as that of the
masking threshold. As a matter of fact, since the audio signal is
normalized in [-1,+1], its nominal PSD is 0 dB.
'''
masking_threshold = psychoacoustical_model(audio_signal[PAM_frame])
shaping_filter_response = masking_threshold
b0, ai = shaping_filter_design(shaping_filter_response, N_coef)

# Plotting results.
plt.figure()
vt.plot_pwelch(audio_signal[PAM_frame], fs=2, new_figure=False,
               label='Audio signal')
W, H = signal.freqz(b0, ai, worN=256)
plt.step(W/np.pi, masking_threshold, 'k--', where='post',
         label='Masking threshold')
plt.plot(W/np.pi, 20*np.log10(np.abs(H)), 'r', linewidth=2,
         label='Filter response')
plt.xlabel('Normalized Frequency : f/(Fs/2)')
plt.legend()

# %%
'''
Applying this perceptual shaping procedure to the whole watermark signal
requires to process the audio signal block per block. Notice that the
filtering continuity from one block to another is ensured by the |state|
vector, which stores the final state of the filter at the end of one block
and applies it as initial conditions for the next block. 
'''
watermark_signal = np.zeros(N_bits*N_samples)
state = np.zeros(N_coef)
n_PAM_frames = (N_bits*N_samples)//N_samples_PAM

for m in range(n_PAM_frames):
    PAM_frame = slice(m*N_samples_PAM, m*N_samples_PAM+N_samples_PAM)

    # Shaping filter design
    masking_threshold = psychoacoustical_model(audio_signal[PAM_frame])
    shaping_filter_response = masking_threshold
    b0, ai = shaping_filter_design(shaping_filter_response, N_coef)

    # Filtering stage
    watermark_signal[PAM_frame], state = \
        signal.lfilter([b0], ai, modulated_signal[PAM_frame], zi=state)

# Filtering the last, incomplete frame in |watermark_signal|
PAM_frame = slice(n_PAM_frames*N_samples_PAM, N_bits*N_samples)
watermark_signal[PAM_frame] = \
    signal.lfilter([b0], ai, modulated_signal[PAM_frame], zi=state)[0]

plt.figure()
plt.plot(np.arange(Fs)/Fs, audio_signal[:Fs], label='audio signal')
plt.plot(np.arange(Fs)/Fs, watermark_signal[:Fs], 'r-',
         label='watermark signal')
plt.xlabel('time (s)')
plt.legend()

# %%
'''
The watermarked audio signal is still obtained by adding the audio signal
and the watermark signal. Plotting again a few samples of the watermark,
audio, and watermarked signals shows that the watermark signal has now
been filtered. As a result, the watermark is quite inaudible, as confirmed
by a listening test. Its level, though, is similar to (if not higher than)
that of the watermark signal in Section 2.
'''
watermarked_signal = watermark_signal + audio_signal

rng = np.arange(34738, 34938)

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(rng/Fs, watermark_signal[rng], 'k')
plt.xlim([rng[0]/Fs, rng[-1]/Fs])
plt.subplot(3, 1, 2)
plt.plot(rng/Fs, audio_signal[rng])
plt.xlim([rng[0]/Fs, rng[-1]/Fs])
plt.subplot(3, 1, 3)
plt.plot(rng/Fs, watermarked_signal[rng], 'r')
plt.xlim([rng[0]/Fs, rng[-1]/Fs])
plt.xlabel('Time(s)')

soundsc(watermarked_signal, Fs)

# %%
'''
3.2 Receiver, based on zero-forcing equalization and detection
The zero-forcing equalization aims at reversing the watermark
perceptual shaping, before extracting the embedded message. The shaping
filter with frequency response |shaping_filter_response| is therefore
recomputed from the audio |watermarked_signal|, since the original
|audio_signal| is not available at the receiver. This process follows the
same block processing as the watermark synthesis, but involves a moving
average filtering stage: filtering the |watermarked_signal| by the filter
whose frequency response is the inverse of the |shaping_filter_response|
yields the filtered received signal denoted by |equalized_signal|. 

Plotting the PSD of the |equalized_signal| shows its flat spectral
envelope (hence its name).
'''
equalized_signal = np.zeros(N_bits*N_samples)
state = np.zeros(N_coef)

for m in range(n_PAM_frames):
    PAM_frame = slice(m*N_samples_PAM, m*N_samples_PAM+N_samples_PAM)

    # Shaping filter design, based on the watermarked signal
    masking_threshold = psychoacoustical_model(watermarked_signal[PAM_frame])
    shaping_filter_response = masking_threshold
    b0, ai = shaping_filter_design(shaping_filter_response, N_coef)

    # Filtering stage
    equalized_signal[PAM_frame], state = \
        signal.lfilter(ai/b0, 1, watermarked_signal[PAM_frame], zi=state)

    if m == 10:
        # Showing the frequency response of the equalizer and PSD of the
        # |equalized_signal|, for frame 10.
        plt.figure()
        vt.plot_pwelch(equalized_signal[PAM_frame], fs=2, new_figure=False,
                       label='Equalized signal')
        W, H = signal.freqz(ai/b0, 1, worN=256)
        plt.plot(W/np.pi, 20*np.log10(np.abs(H)), 'r', linewidth=2,
                 label='Equalizer response')
        plt.xlabel('Normalized Frequency : f/(Fs/2)')
        plt.legend()

# %%
'''
The |equalized_signal| is theoretically the sum of the original watermark
(the |modulated_signal|, which has values in {-1,+1}) and the
|equalized_audio_signal|, which is itself the original |audio_signal|
filtered by the Zero-Forcing equalizer. In other words, from the receiver
point of view, everything looks as if the watermark had been added to the
|equalized_audio_signal| rather than to the |audio_signal| itself. 
Although the |equalized_audio_signal| is not available to the receiver, it
is interesting to compute it and compare it to the original |audio_signal|.
'''
equalized_audio_signal = np.zeros(N_bits*N_samples)
state = np.zeros(N_coef)

for m in range(n_PAM_frames):
    PAM_frame = slice(m*N_samples_PAM, m*N_samples_PAM+N_samples_PAM)

    # Shaping filter design, based on the watermarked signal
    masking_threshold = psychoacoustical_model(watermarked_signal[PAM_frame])
    shaping_filter_response = masking_threshold
    b0, ai = shaping_filter_design(shaping_filter_response, N_coef)

    # Filtering stage
    equalized_audio_signal[PAM_frame], state = \
        signal.lfilter(ai/b0, 1, audio_signal[PAM_frame], zi=state)

# Plotting 200 samples of the |equalized_audio_signal| and
# |modulated_signal|. Notice the level of the |equalized_audio_signal| is
# much higher than that of the original |audio_signal|, mostly because its
# HF content has been enhanced by the equalizer. 
rng = np.arange(34738, 34938)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(rng/Fs, equalized_audio_signal[rng])
plt.xlim([rng[0]/Fs, rng[-1]/Fs])
plt.subplot(2, 1, 2)
plt.plot(rng/Fs, modulated_signal[rng])
plt.xlim([rng[0]/Fs, rng[-1]/Fs])
plt.ylim([-1.1, 1.1])
plt.xlabel('Time(s)')

# %%
'''
The SNR can be estimated from the PSDs of the |equalized_audio_signal|
and |modulated_signal|, or computed easily from the samples of these
signals.
'''
rng = np.arange(34397, 34938)

plt.figure()
vt.plot_pwelch(equalized_audio_signal[rng], fs=2, new_figure=False,
               label='noise')
w, h_dB = vt.pwelch(modulated_signal[rng], fs=2)
plt.plot(w, h_dB, '--r', linewidth=2, label='signal')
plt.xlabel('Normalized Frequency : f/(Fs/2)')
plt.ylim([-30, 50])
plt.legend()

snr_equalized = 10*np.log10(np.var(modulated_signal[rng]) /
                            np.var(equalized_audio_signal[rng]))
print('snr_equalized = %.2f dB' % snr_equalized)

# %%
'''
We can now proceed to the watermark extraction by applying the
correlation demodulator to the |equalized_signal| and compute
the BER. As expected, the obtained BER is quite high, since the level of
the |equalized_audio_signal| is high compared to that of the embedded
|modulated_signal| (in other words, the SNR is low). 
'''
for m in range(N_bits):
    alpha[m] = np.dot(equalized_signal[m*N_samples:m*N_samples+N_samples],
                      spread_waveform)/N_samples
    received_bits[m] = 0 if alpha[m] <= 0 else 1

# Plotting the input bits, the value of alpha, and the received bits.
rng = np.arange(80, 130)

plt.figure()
plt.subplot(3, 1, 1)
plt.step(rng+1, bits[rng], where='post')
plt.ylim([-0.1, 1.1])
plt.subplot(3, 1, 2)
plt.step(rng+1, alpha[rng], where='post')
plt.subplot(3, 1, 3)
plt.step(rng+1, received_bits[rng], where='post')
plt.ylim([-0.1, 1.1])
plt.xlabel('frame index')

number_of_erroneous_bits = np.sum(bits != received_bits)
BER = number_of_erroneous_bits/N_bits
print('number_of_erroneous_bits =', number_of_erroneous_bits)
print('BER =', BER)
print('received_message =', bits_to_message(received_bits))

# %%
'''
3.3 Wiener filtering
The Wiener filtering stage aims at enhancing the SNR between the 
|modulated_signal| and the |equalized_audio_signal|.
This is achieved here by filtering the |equalized_signal| by a
symmetric (non causal) FIR filter with |N_coef|=50 coefficients. 
Its coefficients |hi| are computed so that the output of the filter (when
fed with the |equalized_signal|) becomes maximally similar to the
modulated_signal in the RMSE sense. They are the solution of the
so-called Wiener-Hopf equations:

|hi = inv(equalized_signal_cov_mat)*modulated_signal_autocor_vect|

where |equalized_signal_cov_mat| and |modulated_signal_autocor_vect| are
the covariance matrix of the |equalized_signal| and the autocorrelation
vector of the |modulated_signal|, respectively. 

Since the |modulated_signal| is unknown from the receiver, its
autocorrelation is estimated from an arbitrary modulated signal and can
be computed once. To make it simple, we use our previously computed
|modulated_signal| here. On the contrary, the covariance matrix of the
equalized signal, and therefore the coefficients |hi|, have to be updated
each time the properties of |equalized_signal| change, that is every
|N_samples_PAM|=512 samples.
'''
modulated_signal_autocor_vect = xcorr(modulated_signal, N_coef, 'biased')

Wiener_output_signal = np.zeros(N_bits*N_samples + N_coef)
state = np.zeros(2*N_coef)
hi_78 = None

for m in range(n_PAM_frames):
    PAM_frame = slice(m*N_samples_PAM, m*N_samples_PAM+N_samples_PAM)

    # Estimating the covariance matrix of |equalized_signal|, as a Toeplitz
    # matrix with first row given by the autocorrelation vector of the
    # signal.
    equalized_signal_autocor_vect = xcorr(equalized_signal[PAM_frame],
                                          2*N_coef, 'biased')
    equalized_signal_cov_mat = toeplitz(
        equalized_signal_autocor_vect[2*N_coef:])

    # Estimating the impulse response of the Wiener filter as the solution
    # of the Wiener-Hopf equations.
    hi = np.linalg.solve(equalized_signal_cov_mat,
                         modulated_signal_autocor_vect)

    # Filtering stage
    Wiener_output_signal[PAM_frame], state = \
        signal.lfilter(hi, 1, equalized_signal[PAM_frame], zi=state)
    power = np.linalg.norm(Wiener_output_signal[PAM_frame])**2/N_samples_PAM
    if power != 0:
        Wiener_output_signal[PAM_frame] = \
            Wiener_output_signal[PAM_frame]/np.sqrt(power)

    # Saving the Wiener filter for frame #78
    if m == 78:
        hi_78 = hi/np.sqrt(power)

# Filtering the last, incomplete frame in |equalized_signal|
PAM_frame = slice(n_PAM_frames*N_samples_PAM, N_bits*N_samples)
Wiener_output_signal[PAM_frame] = \
    signal.lfilter(hi, 1, equalized_signal[PAM_frame], zi=state)[0]
power = np.linalg.norm(Wiener_output_signal[PAM_frame])**2/N_samples_PAM
if power != 0:
    Wiener_output_signal[PAM_frame] = \
        Wiener_output_signal[PAM_frame]/np.sqrt(power)

# Since the Wiener filter is non-causal (with |N_coef|=50 coefficients for
# the non-causal part), the resulting |Wiener_output_signal| is delayed
# with |N_coef| samples. 
Wiener_output_signal = np.concatenate((Wiener_output_signal[N_coef:],
                                       np.zeros(N_coef)))

# %%
'''
It is interesting to check how the equalized audio signal and the
modulated signal have been modified by the Wiener filter.
'''
rng = np.arange(34397, 34938)
Wiener_output_audio = signal.lfilter(hi_78, 1, equalized_audio_signal[rng])
Wiener_output_modulated = signal.lfilter(hi_78, 1, modulated_signal[rng])
rng = np.arange(34738, 34938)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(rng/Fs, Wiener_output_audio[340:540])
plt.xlim([rng[0]/Fs, rng[-1]/Fs])
plt.subplot(2, 1, 2)
plt.plot(rng/Fs, Wiener_output_modulated[340:540])
plt.xlim([rng[0]/Fs, rng[-1]/Fs])
plt.xlabel('Time(s)')

# %%
'''
The new SNR can be estimated from the PSDs of the filtered
|equalized_audio_signal| and |modulated_signal|. Obviously, the Wiener
filter has enhanced the frequency bands dominated by the modulated signal,
thereby increasing the SNR.
'''
plt.figure()
vt.plot_pwelch(Wiener_output_audio, fs=2, new_figure=False, label='noise')
w, h_dB = vt.pwelch(Wiener_output_modulated, fs=2)
plt.plot(w, h_dB, '--r', linewidth=2, label='signal')
W, H = signal.freqz(hi_78, 1)
plt.plot(W/np.pi, 20*np.log10(np.abs(H)), ':k', linewidth=2,
         label='Wiener filter response')
plt.xlabel('Normalized Frequency : f/(Fs/2)')
plt.ylim([-60, 30])
plt.legend()

snr_Wiener = 10*np.log10(np.var(Wiener_output_modulated[340:530]) /
                         np.var(Wiener_output_audio[340:530]))
print('snr_Wiener = %.2f dB' % snr_Wiener)

# %%
'''
We can finally apply the correlation demodulator to the estimated
modulated signal. The resulting BER is lower thanks to Wiener filtering. 
It has the same order of magnitude as the one we obtained in Section 1,
while the watermark is now inaudible.
'''
for m in range(N_bits):
    alpha[m] = np.dot(Wiener_output_signal[m*N_samples:m*N_samples+N_samples],
                      spread_waveform)/N_samples
    received_bits[m] = 0 if alpha[m] <= 0 else 1

# Plotting the input bits, the value of alpha, and the received bits.
rng = np.arange(80, 130)

plt.figure()
plt.subplot(3, 1, 1)
plt.step(rng+1, bits[rng], where='post')
plt.ylim([-0.1, 1.1])
plt.subplot(3, 1, 2)
plt.step(rng+1, alpha[rng], where='post')
plt.subplot(3, 1, 3)
plt.step(rng+1, received_bits[rng], where='post')
plt.ylim([-0.1, 1.1])
plt.xlabel('frame index')

number_of_erroneous_bits = np.sum(bits != received_bits)
BER = number_of_erroneous_bits/N_bits
print('number_of_erroneous_bits =', number_of_erroneous_bits)
print('BER =', BER)
print('received_message =', bits_to_message(received_bits))

# %%
'''
3.4 Robustness to MPEG compression
We finally focus on the robustness of our system to MPEG compression.
Using the functions developed in Chapter 3, we can easily apply an mp3
coding/decoding operation to the audio watermarked signal, yielding the
distorted audio watermarked signal |compressed_watermarked_signal|. 

Function involved:

* |output_signal = mp3_codec( input_signal, Fs )| returns the signal
resulting from an mp3 coding/decoding operation applied to |input_signal|,
sampled at |Fs|. (See Chapter 3.)
'''
compressed_watermarked_signal = mp3_codec(watermarked_signal, Fs)

# %%
'''
We then pass the compressed signal through the Zero-Forcing equalizer and
the Wiener filter.
'''
modulated_signal_autocor_vect = xcorr(modulated_signal, N_coef, 'biased')
filtered_signal = np.zeros(N_bits*N_samples)
Wiener_filtered_signal = np.zeros(N_bits*N_samples)
state1 = np.zeros(N_coef)
state3 = np.zeros(2*N_coef)

for m in range(n_PAM_frames):
    PAM_frame = slice(m*N_samples_PAM, m*N_samples_PAM+N_samples_PAM)

    # Zero-Forcing filtering
    masking_threshold = psychoacoustical_model(
        compressed_watermarked_signal[PAM_frame])
    shaping_filter_response = masking_threshold
    b0, ai = shaping_filter_design(shaping_filter_response, N_coef)
    filtered_signal[PAM_frame], state1 = signal.lfilter(
        ai/b0, 1, compressed_watermarked_signal[PAM_frame], zi=state1)

    # Wiener filtering
    filtered_signal_autocor_vect = xcorr(filtered_signal[PAM_frame],
                                         2*N_coef, 'biased')
    filtered_signal_cov_mat = toeplitz(filtered_signal_autocor_vect[2*N_coef:])
    hi = np.linalg.solve(filtered_signal_cov_mat,
                         modulated_signal_autocor_vect)

    Wiener_filtered_signal[PAM_frame], state3 = \
        signal.lfilter(hi, 1, filtered_signal[PAM_frame], zi=state3)
    power = np.linalg.norm(Wiener_filtered_signal[PAM_frame])**2/N_samples_PAM
    if power != 0:
        Wiener_filtered_signal[PAM_frame] = \
            Wiener_filtered_signal[PAM_frame]/np.sqrt(power)

# Filtering the last, incomplete frame in |filtered_signal|
PAM_frame = slice(n_PAM_frames*N_samples_PAM, N_bits*N_samples)
filtered_signal[PAM_frame] = signal.lfilter(
    ai/b0, 1, compressed_watermarked_signal[PAM_frame], zi=state1)[0]
Wiener_filtered_signal[PAM_frame] = signal.lfilter(
    hi, 1, filtered_signal[PAM_frame], zi=state3)[0]

# Wiener delay suppression
Wiener_filtered_signal = np.concatenate(
    (Wiener_filtered_signal[N_coef:N_bits*N_samples], np.zeros(N_coef)))

# %%
'''
We finally apply our correlation detection scheme to this distorted signal
and compute the BER. 
'''
for m in range(N_bits):
    alpha[m] = np.dot(
        Wiener_filtered_signal[m*N_samples:m*N_samples+N_samples],
        spread_waveform)/N_samples
    received_bits[m] = 0 if alpha[m] <= 0 else 1

number_of_erroneous_bits = np.sum(bits != received_bits)
BER = number_of_erroneous_bits/N_bits
print('number_of_erroneous_bits =', number_of_erroneous_bits)
print('BER =', BER)
print('received_message =', bits_to_message(received_bits))

# %%
'''
Unfortunately, the obtained BER has significantly increased: although it
has been shown above that the psychoacoustic model we use in our
perceptual watermarking system is robust to the watermark (i.e., the
masking threshold does not change significantly when the watermark is
added to the audio signal), it is clearly not robust yet to MPEG
compression.  

To show this, let us compare the PSD of the watermarked signal and the
associated masking threshold, before and after the MPEG compression, on
the 11th frame for instance. Clearly, the masking threshold is very
sensitive to MPEG compression for frequencies over 11 kHz. As a result,
the perceptual shaping filter and the equalizer no longer cancel each
other, hence our high BER.
'''
PAM_frame = np.arange(10*N_samples_PAM, 10*N_samples_PAM+N_samples_PAM)
freq_axis = np.arange(1, N_samples_PAM//2+1)*2/N_samples_PAM

plt.figure(figsize=(12, 5))

# Plotting the masking threshold, before MPEG compression
plt.subplot(1, 2, 1)
vt.plot_pwelch(watermarked_signal[PAM_frame], fs=2, new_figure=False,
               label='Audio signal')
masking_threshold = psychoacoustical_model(watermarked_signal[PAM_frame])
plt.step(freq_axis, masking_threshold, 'r', linewidth=2, where='post',
         label='Masking threshold')
plt.xlabel('Normalized Frequency : f/(Fs/2)')
plt.legend()
plt.axis([0, 1, -100, 0])
plt.title('Before MPEG compression')

# Plotting the masking threshold, after MPEG compression
plt.subplot(1, 2, 2)
vt.plot_pwelch(compressed_watermarked_signal[PAM_frame], fs=2,
               new_figure=False, label='Audio signal')
masking_threshold = psychoacoustical_model(
    compressed_watermarked_signal[PAM_frame])
plt.step(freq_axis, masking_threshold, 'r', linewidth=2, where='post',
         label='Masking threshold')
plt.xlabel('Normalized Frequency : f/(Fs/2)')
plt.legend()
plt.axis([0, 1, -100, 0])
plt.title('After MPEG compression')
plt.tight_layout()

# %%
'''
4. Informed watermarking, robust to MPEG compression
In order to improve the robustness of the system exposed in the previous
Section, the watermark information should obviously be spread in the
[0 Hz, 11 kHz] frequency range of the audio signal. This can be achieved
with a low pass filter with cutoff frequency set to 11 kHz. As we shall
see below, this filter will interfere with our watermarking system in
three stages.
'''
# Preparing the data, as in Section 2.
bits = message_to_bits(message)
symbols = bits*2 - 1
N_bits = len(bits)
np.random.seed(12345)
Fs = 44100
R = 100
N_samples = int(round(Fs/R))
SNR_dB = -20
spread_waveform = 2*np.round(np.random.rand(N_samples))-1   # in {-1,1}
modulated_signal = np.zeros(N_bits*N_samples)
for m in range(N_bits):
    modulated_signal[m*N_samples:m*N_samples+N_samples] = \
        symbols[m]*spread_waveform
Fs, audio_raw = wavfile.read('../../audio_samples/violin.wav')
audio_signal = audio_raw[:N_bits*N_samples] / 32768

# %%
'''
4.1 Designing the low pass filter
First, we design a symmetric FIR low pass filter with |Fc=11| kHz cutoff
frequency, using the Parks-McClellan algorithm. Such a filter will have
linear phase, and therefore will not change the shape of the modulated
signal more than required.
We set the order of the filter to |N_coef=50|. Notice that this
filter is non-causal and introduces a |N_coef/2|-samples delay.
'''
Fc = 11000
low_pass_filter = signal.remez(N_coef+1, [0, Fc-1000, Fc+1000, Fs/2],
                               [1, 1E-9], fs=Fs)

# Let us plot the impulse response of this filter.
plt.figure()
plt.plot(low_pass_filter)
plt.xlabel('sample index')

# %%
'''
It is easy to check that its frequency response matches our requirements.
'''
W, H = signal.freqz(low_pass_filter, 1, worN=256, fs=Fs)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(W, 20*np.log10(np.abs(H)+np.finfo(float).tiny))
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(W, np.unwrap(np.angle(H))*180/np.pi)
plt.ylabel('Phase (degrees)')
plt.xlabel('Frequency (Hz)')
plt.grid(True)

# %%
'''
4.2 Modifying the emitter
We first use this low-pass filter to create a spread sequence with
spectral content restricted to the [0 Hz, 11 kHz] range. The modulated  
signal can then be designed as previously. 
'''
spread_waveform = signal.lfilter(
    low_pass_filter, 1, 2*np.round(np.random.rand(N_samples+N_coef))-1)
# Getting rid of the transient
spread_waveform = spread_waveform[N_coef:]
# Making sure the norm of the spread waveform is set to sqrt(N_samples), as
# in the previous Sections.
spread_waveform = spread_waveform / \
    np.sqrt(np.linalg.norm(spread_waveform)**2/N_samples)

for m in range(N_bits):
    modulated_signal[m*N_samples:m*N_samples+N_samples] = \
        symbols[m]*spread_waveform

# %%
'''
To prevent the shaping filter from amplifying the residual frequency
component of the modulated signal over 11 kHz, the
|shaping_filter_response|, initially chosen to be equal to the masking
threshold, is now designed to be equal to the masking threshold in the
[0, 11] kHz frequency band and to zero anywhere else. 
'''
watermark_signal = np.zeros(N_bits*N_samples)
state = np.zeros(N_coef)
N_cutoff = int(np.ceil(Fc/(Fs/2)*(N_samples_PAM/2)))

for m in range(n_PAM_frames):
    PAM_frame = slice(m*N_samples_PAM, m*N_samples_PAM+N_samples_PAM)

    # Shaping filter design
    masking_threshold = psychoacoustical_model(audio_signal[PAM_frame])
    shaping_filter_response = np.concatenate(
        (masking_threshold[:N_cutoff],
         -100*np.ones(N_samples_PAM//2-N_cutoff)))
    b0, ai = shaping_filter_design(shaping_filter_response, N_coef)

    # Filtering stage
    watermark_signal[PAM_frame], state = \
        signal.lfilter([b0], ai, modulated_signal[PAM_frame], zi=state)

# Filtering the last, incomplete frame in |watermark_signal|
PAM_frame = slice(n_PAM_frames*N_samples_PAM, N_bits*N_samples)
watermark_signal[PAM_frame] = \
    signal.lfilter([b0], ai, modulated_signal[PAM_frame], zi=state)[0]

watermarked_signal = audio_signal + watermark_signal

soundsc(watermarked_signal, Fs)

# %%
'''
MPEG compression is then applied to the new |watermarked_signal|.
'''
compressed_watermarked_signal = mp3_codec(watermarked_signal, Fs)

# %%
'''
4.3 Modifying the receiver
Before starting the watermark extraction, the
|compressed_watermarked_signal| is low-passed, to avoid residual 
components over 11 kHz.
'''
compressed_watermarked_signal = signal.lfilter(
    low_pass_filter, 1,
    np.concatenate((compressed_watermarked_signal, np.zeros(N_coef//2))))
# Compensating for the |N_coef/2| delay
compressed_watermarked_signal = compressed_watermarked_signal[N_coef//2:]

# %%
'''
Watermark extraction is finally obtained as in Section 3, except that the
LP filter is still taken into account in the zero-forcing equalization. 
The resulting BER is now similar to the one we obtained with no MPEG
compression. The watermarking system is thus now robust to MPEG
compression.
'''
modulated_signal_autocor_vect = xcorr(modulated_signal, N_coef, 'biased')
filtered_signal = np.zeros(N_bits*N_samples)
Wiener_filtered_signal = np.zeros(N_bits*N_samples)
state1 = np.zeros(N_coef)
state3 = np.zeros(2*N_coef)

for m in range(n_PAM_frames):
    PAM_frame = slice(m*N_samples_PAM, m*N_samples_PAM+N_samples_PAM)

    # Zero-Forcing filtering
    masking_threshold = psychoacoustical_model(
        compressed_watermarked_signal[PAM_frame])
    shaping_filter_response = np.concatenate(
        (masking_threshold[:N_cutoff],
         -100*np.ones(N_samples_PAM//2-N_cutoff)))
    b0, ai = shaping_filter_design(shaping_filter_response, N_coef)

    filtered_signal[PAM_frame], state1 = signal.lfilter(
        ai/b0, 1, compressed_watermarked_signal[PAM_frame], zi=state1)

    # Wiener equalization
    filtered_signal_autocor_vect = xcorr(filtered_signal[PAM_frame],
                                         2*N_coef, 'biased')
    filtered_signal_cov_mat = toeplitz(filtered_signal_autocor_vect[2*N_coef:])
    hi = np.linalg.solve(filtered_signal_cov_mat,
                         modulated_signal_autocor_vect)

    Wiener_filtered_signal[PAM_frame], state3 = \
        signal.lfilter(hi, 1, filtered_signal[PAM_frame], zi=state3)
    power = np.linalg.norm(Wiener_filtered_signal[PAM_frame])**2/N_samples_PAM
    if power != 0:
        Wiener_filtered_signal[PAM_frame] = \
            Wiener_filtered_signal[PAM_frame]/np.sqrt(power)

# Filtering the last, incomplete frame
PAM_frame = slice(n_PAM_frames*N_samples_PAM, N_bits*N_samples)
filtered_signal[PAM_frame] = signal.lfilter(
    ai/b0, 1, compressed_watermarked_signal[PAM_frame], zi=state1)[0]
Wiener_filtered_signal[PAM_frame] = signal.lfilter(
    hi, 1, filtered_signal[PAM_frame], zi=state3)[0]
power = np.linalg.norm(Wiener_filtered_signal[PAM_frame])**2/N_samples_PAM
if power != 0:
    Wiener_filtered_signal[PAM_frame] = \
        Wiener_filtered_signal[PAM_frame]/np.sqrt(power)

Wiener_filtered_signal = np.concatenate(
    (Wiener_filtered_signal[N_coef:N_bits*N_samples], np.zeros(N_coef)))

# Correlation demodulator
for m in range(N_bits):
    alpha[m] = np.dot(
        Wiener_filtered_signal[m*N_samples:m*N_samples+N_samples],
        spread_waveform)/N_bits
    received_bits[m] = 0 if alpha[m] <= 0 else 1

number_of_erroneous_bits = np.sum(bits != received_bits)
total_number_of_bits = N_bits
BER = number_of_erroneous_bits/N_bits
print('number_of_erroneous_bits =', number_of_erroneous_bits)
print('total_number_of_bits =', total_number_of_bits)
print('BER =', BER)
print('received_message =', bits_to_message(received_bits))

# %%
