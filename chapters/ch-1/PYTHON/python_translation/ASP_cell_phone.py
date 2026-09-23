# %%
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import pitch
import find_Nbest_components as fNbc
import lpc_brute_force_calculation as lpc_bfc
import visual_tools as vt

# %%
'''
Chapter 1 - How is speech processed in a cell phone conversation?
This is a companion file to the book "Applied Signal Processing",
by T. Dutoit and F. Marques, Springer 2008.
'''
# %%
'''
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
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white' 

# %%
'''
1. Examining a speech file
Let us load file 'speech.wav', listen to it, and plot its samples. This
file contains the sentence "Paint the circuits" sampled at 8 kHz, with 16
bits. (This sentence was taken from the Open Speech Repository on the
web)
'''
sample_rate, audio_raw = wavfile.read("../../audio_samples/speech.wav")
audio = audio_raw / 32768
total_duration = len(audio)
speech_time  = np.arange(0, total_duration, 1)
y_tick_marks = np.arange(-0.6, 0.7, 0.1)
x_tick_marks = np.arange(0,10000, 1000)
plt.figure  (figsize=(10, 8))
plt.xlabel  ("Time (samples)")
plt.ylabel  ("Amplitude")
plt.yticks  (y_tick_marks)
plt.xticks  (x_tick_marks)
plt.plot    (speech_time, audio)
plt.tick_params(axis='both', direction='in',right=True, top=True)

silence = np.zeros(int(2000), dtype=audio_raw.dtype)
audio_raw_padded = np.concatenate((audio_raw, silence))
sd.play(audio_raw_padded,8000)
sd.wait()

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
vt.plot_spectrogram(audio)

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
sample_rate, audio_raw = wavfile.read("../../audio_samples/speech.wav")
audio = audio_raw / 32768
input_frame_for_letter_e = audio[3499:3739]
plt.figure  (figsize=(10, 8))
plt.plot    (input_frame_for_letter_e)
plt.title   ('Zooming on the letter "e"')
plt.xlabel  ('Time (samples)')
plt.ylabel  ('Amplitude')

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
vt.plot_periodogram(input_frame_for_letter_e, 2*np.pi)

# %%
'''
The fundamental frequency appears again at around 125 Hz. One can also
roughly estimate the position of formants (peaks in the spectral
envelope) at +- 300 Hz, 1400 Hz, 2700 Hz.
'''
# %%
'''
Let us now fit an LP model of order 10 to our voiced frame. We obtain
the prediction coefficients (ai) and the variance of the residual signal
(sigma_square).
Notice we do not apply windowing prior to LP analysis now, as it has 
no tutorial benefit. We will add it in subsequent Sections.
'''
a_coefficients, sigma_squared = lpc_bfc.lpc_calculation(input_frame_for_letter_e, 10)
sigma = np.sqrt(sigma_squared)

# %%
'''
The estimation algorithm inside LPC uses a brute force method. It chooses the coefficients of an FIR filter A(z) so that when
passing the input frame into A(Z), the output, termed as the prediction
residual, has minimum energy. It can be shown that this leads to a filter
which has anti-resonances wherever the input frame has a formant. For
this reason, the A(z) filter is termed as the "inverse" filter. Let us
plot its frequency response (on 512 points), and superimpose it to that
of the "synthesis" filter 1/A(z).
'''
W, H = signal.freqz(1, a_coefficients, worN=512)
WI, HI = signal.freqz(a_coefficients, 1, worN=512)
plt.figure(figsize=(10, 8))
x_tick_marks = np.arange(0, 1.1, 0.1)
plt.plot    (W/np.pi, 20*np.log10(np.abs(H)),'-', 
             label='Synthesis filter 1/A(z)')
plt.plot    (WI/np.pi, 20*np.log10(np.abs(HI)),'--', 
             label='Inverse filter A(z)')
plt.xlabel  ('Normalized frequency ($\\times \\pi$ rad/sample)')
plt.ylabel  ('Magnitude (dB)')
plt.xticks  (x_tick_marks)
plt.legend  ()

# %%
'''
In other words, the frequency response of the filter 1/A(z) matches the
spectral amplitude envelope of the frame. Let us superimpose this
frequency response to periodogram of the vowel.
'''
vt.plot_periodogram(input_frame_for_letter_e, 2)
plt.plot(W/np.pi,20*np.log10(sigma*abs(H)))

# %%
'''
Formants appear again, much better than before. As a matter of fact, LP
modelling acts as a parametric spectral estimator, resulting in a smooth
estimate.

In other words, the LPC fit has automatically adjusted the poles of the
synthesis filter close to the unit circle at angular positions chosen to
imitate formant resonances.
'''
vt.plot_zplane([1], a_coefficients)

# %%
'''
If we apply the inverse of this filter to the input frame, we obtain the
prediction residual.
'''
def plot_filter(b,a,x):
    filter_output = signal.lfilter(b, a, x)
    plt.figure(figsize=(10, 8))
    plt.plot(filter_output, color='blue')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    return filter_output

LP_residual = plot_filter(a_coefficients,[1],input_frame_for_letter_e)

# %%
'''
Let us compare the spectrum of this residual to the original spectrum.
The new spectrum is approximately flat; its fine spectral details,
however, are the same as those of the analysis frame. In particular, its
pitch and harmonics are preserved. 
'''
vt.plot_periodogram(LP_residual, 2*np.pi)

# %%
'''
For obvious reasons, applying the synthesis filter to this prediction
residual results in the analysis frame itself (since the synthesis filter
is the inverse of the inverse filter).
'''
output_frame = plot_filter([1], a_coefficients, LP_residual)

# %%
'''
The LPC model actually models the prediction residual of voiced speech as
an impulse train with adjustable pitch period and amplitude. For the
speech frame considered, for instance, the LPC excitation is a sequence
of pulses separated by 64 zeros (so as to impose a period of 65 samples).
Notice we multiply the excitation by some gain so that its variance
matches that of the residual signal. 
'''
excitation = np.zeros(240)
excitation[::65] = 1
gain = sigma / np.sqrt(1/65)
plt.figure(figsize=(10, 8))
plt.plot(gain*excitation)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')

# %%
'''
Clearly, as far as the waveform is concerned, the LPC excitation is far
from similar to the prediction residual. Its spectrum, however, has the
same broad features as that of the residual: flat envelope, and harmonic
content corresponding to F0. The main difference is that the excitation
spectrum is "over-harmonic" compared to the residual spectrum.
'''
vt.plot_periodogram(gain*excitation, 2*np.pi)

# %%
'''
Let us now use the synthesis filter to produce an artificial "e". 
'''
synt_frame = plot_filter(gain, a_coefficients, excitation)

# %%
'''
Although the resulting waveform is obviously different from the original
one (this is due to the fact that the LP model does not account for the
phase spectrum of the original signal), its spectral envelope is
identical. Its fine harmonic details, though, also widely differ (the
synthetic frame is actually "over-harmonic" compared to the analysis
frame.
'''
vt.plot_periodogram(synt_frame, 2*np.pi)

# %%
'''
3. Linear prediction synthesis of 30 ms of unvoiced speech
It is easy to apply the same process to an unvoiced frame, and compare
the final spectra again. Let us first extract an unvoiced frame and plot
it.
'''
sample_rate, audio_raw = wavfile.read("../../audio_samples/speech.wav")
audio = audio_raw / 32768
input_frame_for_letter_c = audio[4499:4739]
plt.figure  (figsize=(10, 8))
plt.plot    (input_frame_for_letter_c)
plt.title   ('Zooming on the letter "c"')
plt.xlabel  ('Time (samples)')
plt.ylabel  ('Amplitude')

# %%
'''
As expected, no clear periodicity appears.

Now let us see the spectral content of this speech frame. Notice that,
since we are dealing with noisy signals, we use the averaged periodogram
to have a better estimate power spectral densities, although with less
frequency resolution than using a simple periodogram.
'''
vt.plot_welch(input_frame_for_letter_c)

# %%
'''
Let us now apply an LP model of order 10, and synthesize a new frame.
Synthesis is performed by all-pole filtering a Gaussian white noise frame
with standard deviation set to the prediction residual standard
deviation, sigma.
'''
a_coefficients, sigma_squared = lpc_bfc.lpc_calculation(input_frame_for_letter_c, 10)
sigma = np.sqrt(sigma_squared)
excitation = np.random.randn(240)
synt_frame = plot_filter(sigma,a_coefficients,excitation)

# %%
'''
The synthetic waveform has no sample in common with the original
waveform.
'''
# %%
'''
The spectral envelope of this frame, however, is very similar to the
original one.
'''
vt.plot_welch(synt_frame)

# %%
'''
4. Linear prediction synthesis of a speech file, with fixed F0
We will now loop the previous operations for the complete speech file,
using 30ms analysis frames overlapping by 20 ms. Frames are now weighted
by a Hamming window. At synthesis time, we simply synthesize 10 ms of
speech, and concatenate the resulting synthetic frames to obtain the
output speech file. Let us choose 200 Hz as synthesis F0, for
convenience: this way each 10ms excitation frame contains exactly two
pulses.
'''
sample_rate, audio_raw = wavfile.read("../../audio_samples/speech.wav")
audio = audio_raw/32768
synt_speech_V = []

for i in range(int((len(audio)-160)/80)): # number of frames

    # Extracting the analysis frame
    input_frame = audio[i*80:i*80+240]

    # Hamming window weighting
    windowed_frame = input_frame*np.hamming(240)
    a_coefficients, sigma_squared = lpc_bfc.lpc_calculation(windowed_frame, 10)
    sigma = np.sqrt(sigma_squared)

    # Generating 10 ms of excitation
    # = 2 pitch periods at 200 Hz
    excitation = np.zeros(80)
    excitation[::40] = 1
    gain = sigma/np.sqrt(1/40)

    # Applying the synthesis filter
    synt_frame = signal.lfilter(gain, a_coefficients, excitation)

    # Concatenating synthesis frames
    synt_speech_V.extend(synt_frame)

synt_speech_V = np.array(synt_speech_V)
print(synt_speech_V)
plt.figure  (figsize=(10,8))
plt.plot    (synt_speech_V)
plt.title   ('Synthesized Speech Waveform (F0 = 200Hz)')
plt.xlabel  ('Time (samples)')
plt.ylabel  ('Amplitude')
plt.grid    (True)

# %%
'''
The output waveform basically contains a sequence of LP filter impulse
responses. Let us zoom on 30 ms of LPC speech.
'''
plt.figure  (figsize=(10,8))
plt.plot    (synt_speech_V[3299:3539])
plt.title   ('Zooming on 30ms of LPC speech')
plt.xlabel  ('Time (samples)')
plt.ylabel  ('Amplitude')
plt.grid    (True)

# %%
'''
It appears that in many cases they impulse responses have been cropped.
As a matter of fact, since each synthesis frame was composed of two
identical impulses, one should expect our LPC speech to exhibit pairs of
identical pitch periods. This is not the case, due to the fact that for
producing each new synthetic frame the internal variables of the
synthesis filter are implicitly reset to zero. We can avoid this problem
by maintaining the internal variables of the filter from the end of each
frame to the beginning of the next one.
'''
sample_rate, audio_raw = wavfile.read("../../audio_samples/speech.wav")
audio = audio_raw/32768
synt_speech_V_zi = []

z = np.zeros(10) # internal variables of the synthesis filter

for i in range(int((len(audio)-160)/80)):

    input_frame = audio[i*80:i*80+240]
    windowed_frame = input_frame*np.hamming(240)
    a_coefficients, sigma_squared = lpc_bfc.lpc_calculation(windowed_frame, 10)
    sigma = np.sqrt(sigma_squared)

    # Generating 10 ms of excitation
    excitation = np.zeros(80)
    excitation[::40] = 1
    gain = sigma/np.sqrt(1/40)

    # Applying the synthesis filter
    # Taking care of the internal variables of the filter
    synt_frame, z = signal.lfilter([gain], a_coefficients, excitation, zi=z)

    synt_speech_V_zi.extend(synt_frame)

synt_speech_V_zi = np.array(synt_speech_V_zi)
plt.figure  (figsize=(10,8))
plt.plot    (synt_speech_V_zi)
plt.title   ('Synthesized Speech (smoothed)')
plt.xlabel  ('Time (samples)')
plt.ylabel  ('Amplitude')
plt.grid    (True)

silence = np.zeros(int(2000), dtype=synt_speech_V_zi.dtype)
synt_speech_V_zi_padded = np.concatenate((synt_speech_V_zi, silence))
sd.play(synt_speech_V_zi_padded,8000)
sd.wait()

# %%
'''
This time the end of each impulse response is properly added to the
beginning of the next one, which results in more smoothly evolving
periods.
'''
plt.figure  (figsize=(10,8))
plt.plot    (synt_speech_V_zi[3299:3539])
plt.title   ('Zooming on 30ms of LPC speech (zi filter memory applied)')
plt.xlabel  ('Time (samples)')
plt.ylabel  ('Amplitude')
plt.grid    (True)

# %%
'''
If we want to synthesize speech with constant pitch period length
different from a sub-multiple of 80 samples (say, 70 samples), we
additionally need to take care of a possible pitch period offset in the
excitation signal.
'''
sample_rate, audio_raw = wavfile.read("../../audio_samples/speech.wav")
audio = audio_raw/32768
synt_speech_V_offset = []
z = np.zeros(10)

offset = 0  # Offset of the next pitch pulse with respect to the 
            # start of the current frame.

N0 = 65  # Constant synthesis pitch period (in samples)

for i in range(int((len(audio)-160)/80)):

    input_frame = audio[i*80:i*80+240]
    windowed_frame = input_frame*np.hamming(240)
    a_coefficients, sigma_squared = lpc_bfc.lpc_calculation(windowed_frame, 10)
    sigma = np.sqrt(sigma_squared)

    # Generating 10 ms of excitation
    # taking a possible offset into account

    # if pitch period length > excitation frame length
    if offset >= 80:
        excitation = np.zeros(80)
        offset = offset-80
    else:
        # complete the previously unfinished pitch period
        excitation = np.zeros(offset)
        # for all pitch periods in the remaining of the frame
        for j in range(int(np.floor((80-offset)/N0))):
            # add one excitation period
            excitation = np.concatenate((excitation,[1],np.zeros(N0-1)),axis=None) 
        # number of samples left in the excitation frame
        flush = int(80-len(excitation))
        if flush!=0: 
            # fill the frame with a partial pitch period
            excitation =  np.concatenate((excitation,[1],np.zeros(flush-1)),axis=None) 
            # remember to fill the remaining of the period in next frame 
            offset = N0-flush
        else:
            offset = 0

    gain=sigma/np.sqrt(1/N0)
    synt_frame, z = signal.lfilter([gain], a_coefficients, excitation, zi=z)
    synt_speech_V_offset.extend(synt_frame)

synt_speech_V_offset = np.array(synt_speech_V_offset)
plt.figure  (figsize=(10,8))
plt.plot    (synt_speech_V_offset)
plt.title   ('Synthesized Speech (offset)')
plt.xlabel  ('Time (samples)')
plt.ylabel  ('Amplitude')
plt.grid    (True)

silence = np.zeros(int(2000), dtype=synt_speech_V_offset.dtype)
synt_speech_V_offset_padded = np.concatenate((synt_speech_V_offset, silence))
sd.play(synt_speech_V_offset_padded,8000)
sd.wait()

# %%
'''
5. Unvoiced linear prediction synthesis of a speech file
Synthesizing the complete speech file as LPC unvoiced speech is easy.
'''
sample_rate, audio_raw = wavfile.read("../../audio_samples/speech.wav")
audio = audio_raw/32768
synt_speech_UV = []
z = np.zeros(10)

for i in range(int((len(audio)-160)/80)):

    input_frame = audio[i*80:i*80+240]
    windowed_frame = input_frame*np.hamming(240)
    a_coefficients, sigma_squared = lpc_bfc.lpc_calculation(windowed_frame, 10)
    sigma = np.sqrt(sigma_squared)

    # Generating 10 ms of excitation
    excitation = np.random.randn(80) # White Gaussian noise
    gain = sigma

    synt_frame, z = signal.lfilter([gain], a_coefficients, excitation, zi=z)
    synt_speech_UV.extend(synt_frame)

synt_speech_UV = np.array(synt_speech_UV)
plt.figure  (figsize=(10,8))
plt.plot    (synt_speech_UV)
plt.title   ('Synthesized Speech (unvoiced)')
plt.xlabel  ('Time (samples)')
plt.ylabel  ('Amplitude')
plt.grid    (True)

silence = np.zeros(int(2000), dtype=synt_speech_UV.dtype)
synt_speech_UV_padded = np.concatenate((synt_speech_UV, silence))
sd.play(synt_speech_UV_padded,8000)
sd.wait()

# %%
'''
6. Linear prediction synthesis of a speech file, with original F0
We will now synthesize the same speech, using the original F0. We will
thus have to deal with the additional problems of pitch estimation (on a
frame-by-frame basis), including voiced/unvoiced decision. This approach
is similar to the LPC10 that of the coder (except we do not quantize
coefficients here).

*Python function involved:*
 
* |T0 = pitch.pitch(input_frame)| : returns the pitch period T0 (in samples) of
a speech frame (T0 is set to zero when the frame is detected as
unvoiced). T0 is obtained from the maximum of the (estimated)
autocorrelation of the LPC residual. Voiced/unvoiced decision is based on
the ratio of this maximum by the variance of the residual.
This simple algorithm is not optimal, but will do the job for this
proof of concept.
'''
sample_rate, audio_raw = wavfile.read("../../audio_samples/speech.wav")
audio = audio_raw/32768
synt_speech_LPC10 = []
z = np.zeros(10)
offset = 0

for i in range(int((len(audio)-160)/80)):

    input_frame = audio[i*80:i*80+240]
    windowed_frame = input_frame*np.hamming(240)
    a_coefficients, sigma_squared = lpc_bfc.lpc_calculation(windowed_frame, 10)
    sigma = np.sqrt(sigma_squared)

    # local synthesis pitch period (in samples)
    N0 = pitch.pitch(input_frame)

    # Generating 10 ms of excitation
    if N0!=0: # voiced frame
        # Generate 10 ms of voiced excitation
        # taking a possible offset into account
        if offset>=80:
            excitation = np.zeros(80)
            offset = offset-80
        else:
            excitation = np.zeros(offset)
            for j in range(int(np.floor((80-offset)/N0))):
                excitation = np.concatenate((excitation,[1],np.zeros(N0-1)),axis=None)
            flush = 80-len(excitation)
            if flush!=0: 
                excitation = np.concatenate((excitation,[1],np.zeros(flush-1)),axis=None)
                offset = N0-flush
            else:
                offset = 0
        gain = sigma/np.sqrt(1/N0)
    else:
        # Generate 10 ms of unvoiced voiced excitation
        excitation = np.random.randn(80) # White Gaussian noise
        gain = sigma
        offset = 0 # reset for subsequent voiced frames  
    
    synt_frame, z = signal.lfilter([gain], a_coefficients,excitation, zi=z)
    synt_speech_LPC10.extend(synt_frame)

synt_speech_LPC10 = np.array(synt_speech_LPC10)
plt.figure  (figsize=(10,8))
plt.plot    (synt_speech_LPC10)
plt.title   ('Synthesized Speech (Original F0)')
plt.xlabel  ('Time (samples)')
plt.ylabel  ('Amplitude')
plt.grid    (True)

silence = np.zeros(int(2000), dtype=synt_speech_LPC10.dtype)
synt_speech_LPC10_padded = np.concatenate((synt_speech_LPC10, silence))
sd.play(synt_speech_LPC10_padded,8000)
sd.wait()

# %%
'''
The resulting synthetic speech is intelligible. It shows the same
formants as the original speech. It is therefore acoustically similar to
the original, except for the additional buzzyness which has been added by
the LP model.
'''
vt.plot_spectrogram(synt_speech_LPC10)

# %%
'''
It is easy to estimate the total bit-rate corresponding to this
proof-of-concept: 42 bits are required for inaudible quantization of the
prediction coefficients. Adding 7 bits for pitch and V/UV and 5 bits for
gain gives 54 bits every 10ms: 5400 bits/s. LPC10 was normalized at 2400
bits/s, which was achieved by using larger synthesis frames (22.5 ms).
'''
# %%
'''
7. CELP analysis-synthesis  of a speech file
Our last step will be to replace the LPC10 excitation by a more realistic
Code-Excited Linear Prediction (CELP) excitation, obtained by
selecting the best linear combination of excitation components from a
codebook. Component selection is performed in a closed loop, so as to
minimize the difference between the synthetic and original signals.
We start with 30 ms LP analysis frames, shifted every 5 ms, and a
codebook size of 512 vectors, from which 10 components are chosen for
every 5 ms synthesis frame.

*Python function involved:*

* |gains, indices = fNbc.find_Nbest_components(sig, codebook_filt, N_components)|

This function finds the N best components of signal from the 
vectors in codebook_vectors, so that the residual error:
    error = signal - codebook_vectors[indices]*gains 
is minimized.
Components are found one-by-one using a greedy algorithm. 
When components in codebook_vectors are not orthogonal, 
the search is therefore suboptimal. 
'''
frame_length = 240   # length of the LPC analysis frame
frame_shift = 40     # length of the excitation and synthesis frames
codebook_size = 512  # number of vectors in the codebook
N_components = 10    # number of codebook components per frame

sample_rate, audio_raw = wavfile.read("../../audio_samples/speech.wav")
audio = audio_raw/32768

# Initializing internal variables
z_inv = np.zeros(10)  # inverse filter
z_synt = np.zeros(10) # synthesis filter
synt_speech_CELP = []

# Generating the stochastic excitation codebook
codebook = np.random.randn(frame_shift,codebook_size)

for i in range(int((len(audio)-frame_length+frame_shift)/frame_shift)):

    input_frame = audio[i*frame_shift : i*frame_shift+frame_length]

    # LPC analysis of order 10
    windowed_frame = input_frame*np.hamming(frame_length)
    a_coefficients, sigma_squared = lpc_bfc.lpc_calculation(windowed_frame, 10)
    
    # Extracting frame_shift samples from the LPC analysis frame
    speech_frame = input_frame[int((frame_length-frame_shift)/2):int((frame_length-frame_shift)/2+frame_shift)]
    
    # Filtering the codebook (all column vectors)
    codebook_filt  = signal.lfilter(1, a_coefficients, codebook, axis=0)
    
    # Finding speech_frame components in the filtered codebook
    # taking into account the transient stored in the internal variables of
    # the synthesis filter
    ringing,last_situation  = signal.lfilter(1, a_coefficients, np.zeros(frame_shift), zi=z_synt)
    sig = speech_frame - ringing
    gains, indices = fNbc.find_Nbest_components(sig, codebook_filt, N_components)
    
    # Generating the corresponding excitation as a weighted sum of
    # codebook vectors
    excitation = np.dot(codebook[:,indices],gains)
    
    # Synthesizing CELP speech, and keeping track of the synthesis filter 
    # internal variables
    synt_frame, z_synt = signal.lfilter(1, a_coefficients, excitation, zi=z_synt)
    synt_speech_CELP.extend(synt_frame)
   
    # Screen output
    if (i + 1) % 10 == 0:
        print(f'frame {i + 1:3d}')

    LP_residual, z_inv = signal.lfilter(a_coefficients, 1, speech_frame, zi=z_inv)
    if i == 139:
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(LP_residual, label='LPC residual')
        plt.plot(excitation, '-.', label='CELP excitation')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.yticks(np.arange(-0.2,0.2,0.05))
        plt.legend(loc='upper right')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(speech_frame, label='original speech')
        plt.plot(synt_frame, '-.', label='synthetic speech')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.yticks(np.arange(-0.6,0.6,0.2))
        plt.legend(loc='upper right')
        plt.grid(True)
       
        plt.tight_layout()

synt_speech_CELP = np.array(synt_speech_CELP)

silence = np.zeros(int(2000), dtype=synt_speech_CELP.dtype)
synt_speech_CELP_padded = np.concatenate((synt_speech_CELP, silence))
sd.play(synt_speech_CELP_padded,8000)
sd.wait()

# %%
'''
The resulting synthetic speech sounds more natural than in LPC10. 
Plosives are much better rendered, and voiced sounds are no longer buzzy,
but speech sounds a bit noisy.
Notice that pitch and V/UV estimation are no longer required. 

One can see that the closed loop optimization leads to excitation frames
which can somehow differ from the LP residual, while the resulting
synthetic speech is more similar to its original counterpart.
'''
# %%
'''
In the above script, though, each new frame was processed independently of
past frames. Since voiced speech is strongly self-correlated, it makes sense
to incorporate in long-term prediction filter in cascade with the LPC
(short-term) prediction filter. In the example below, we can reduce the
number of stochastic components from 10 to 5, while still increasing
speech quality thanks to long-term prediction. 
'''
frame_length    = 240    # length of the LPC analysis frame
frame_shift     = 40     # length of the excitation and synthesis frames
codebook_size   = 512    # number of vectors in the codebook
N_components    = 5      # number of codebook components per frame
LTP_max_delay   = 256    # maximum long-term prediction delay (in samples)

sample_rate, audio_raw = wavfile.read("../../audio_samples/speech.wav")
audio = audio_raw/32768

# Initializing internal variables
z_inv=np.zeros(10)  # inverse filter
z_synt=np.zeros(10) # synthesis filter
synt_speech_CELP_LTP = []
excitation_buffer=np.zeros(LTP_max_delay+frame_shift)

# Building the stochastic excitation codebook. The following line is
# commented, so as to re-use the previous codebook, for comparing the
# output with and without long-term prediction

# codebook = np.random.randn(frame_shift,codebook_size)

for i in range(int((len(audio)-frame_length+frame_shift)/frame_shift)):
    
    input_frame = audio[i*frame_shift : i*frame_shift+frame_length]

    # LPC analysis of order 10
    windowed_frame = input_frame*np.hamming(frame_length)
    a_coefficients, sigma_squared = lpc_bfc.lpc_calculation(windowed_frame, 10)
    sigma = np.sqrt(sigma_squared)
    
    # Extracting frame_shift samples from the LPC analysis frame
    speech_frame = input_frame[int((frame_length-frame_shift)/2):int((frame_length-frame_shift)/2+frame_shift)]
        
    # Building the long-term prediction codebook and filtering it
    LTP_codebook = np.zeros((frame_shift, LTP_max_delay))
    for j in range(LTP_max_delay):
         LTP_codebook[:,j] = excitation_buffer[j:j+frame_shift]
    LTP_codebook_filt = signal.lfilter(1, a_coefficients, LTP_codebook, axis=0)
        
    # Filtering the stochastic codebook (all column vectors)
    codebook_filt = signal.lfilter(1, a_coefficients, codebook, axis=0)
    
    # Finding the best predictor in the LTP codebook
    ringing, x = signal.lfilter(1, a_coefficients, np.zeros(frame_shift), zi=z_synt)
    sig = speech_frame - ringing
    LTP_gain, LTP_index = fNbc.find_Nbest_components(sig, LTP_codebook_filt, 1)
    
    # Generating the corresponding prediction
    LT_prediction = np.dot(LTP_codebook[:,LTP_index],LTP_gain)
        
    # Finding speech_frame components in the filtered codebook
    # taking long term prediction into account 
    sig = sig - np.dot(LTP_codebook_filt[:,LTP_index],LTP_gain)
    gains, indices = fNbc.find_Nbest_components(sig,codebook_filt, N_components)
    
    # Generating the corresponding excitation as a weighted sum of
    # codebook vectors plus long-term prediction
    excitation = LT_prediction + np.dot(codebook[:,indices],gains)
    
    # Synthesizing CELP speech, and keeping track of the synthesis filter 
    # internal variables
    synt_frame, z_synt = signal.lfilter(1, a_coefficients, excitation, zi=z_synt)
    synt_speech_CELP_LTP.extend(synt_frame)
   
    # Updating the excitation buffer for long-term prediction
    excitation_buffer[:LTP_max_delay]=excitation_buffer[frame_shift:LTP_max_delay+frame_shift]
    excitation_buffer[LTP_max_delay:LTP_max_delay+frame_shift]=excitation

    # Screen output
    if (i + 1) % 10 == 0:
        print(f'frame {i + 1:3d}')

    LP_residual, z_inv = signal.lfilter(a_coefficients, 1, speech_frame, zi=z_inv)
    if i==139:
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(LP_residual, label='LPC residual')
        plt.plot(excitation, '-.', label='CELP excitation')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.yticks(np.arange(-0.2,0.2,0.05))
        plt.legend(loc='upper right')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(speech_frame, label='original speech')
        plt.plot(synt_frame, '-.', label='synthetic speech')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.yticks(np.arange(-0.6,0.6,0.2))
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()

synt_speech_CELP_LTP = np.array(synt_speech_CELP_LTP)

silence = np.zeros(int(2000), dtype=synt_speech_CELP_LTP.dtype)
synt_speech_CELP_LTP_padded = np.concatenate((synt_speech_CELP_LTP, silence))
sd.play(synt_speech_CELP_LTP_padded,8000)
sd.wait()

# %%
'''
The resulting synthetic speech is still similar to the original one,
notwithstanding the reduction of the number of stochastic components.  
'''
# %%
'''
While the search for the best components in the previous scripts aims at 
minimizing the energy of the difference between original and synthetic
speech samples, it makes sense to use the fact that the ear will be more
tolerant to this difference in parts of the spectrum that are louder and vice
versa. This can be achieved by applying a perceptual filter to
the error, which enhances spectral components of the error in frequency
bands with less energy, and vice-versa. 
In the following example, we still decrease the number of components
from 5 to 2, with the same overall synthetic speech quality.
'''
frame_length    = 240    # length of the LPC analysis frame
frame_shift     = 40     # length of the excitation and synthesis frames
codebook_size   = 512    # number of vectors in the codebook
N_components    = 2      # number of codebook components per frame
LTP_max_delay   = 256    # maximum long-term prediction delay (in samples)
gamma           = 0.8    # perceptual factor

sample_rate, audio_raw = wavfile.read("../../audio_samples/speech.wav")
audio = audio_raw/32768

# Initializing internal variables
z_inv       =   np.zeros(10) # inverse filter
z_synt      =   np.zeros(10) # synthesis filter
z_gamma_s   =   np.zeros(10) # perceptual filter applied to speech
z_gamma_e   =   np.zeros(10) # perceptual filter applied to excitation
synt_speech_CELP_LTP_PF = []
excitation_buffer= np.zeros(LTP_max_delay+frame_shift)

# Building the stochastic excitation codebook. The following line is
# commented, so as to re-use the previous codebook, for comparing the
# output with and without perceptual filtering

# codebook = np.random.randn(frame_shift,codebook_size)

for i in range(int((len(audio)-frame_length+frame_shift)/frame_shift)):
    
    input_frame = audio[i*frame_shift:i*frame_shift+frame_length]

    # LPC analysis of order 10
    windowed_frame = input_frame*np.hamming(frame_length)
    a_coefficients, sigma_squared = lpc_bfc.lpc_calculation(windowed_frame, 10)
    sigma = np.sqrt(sigma_squared)
    
    # Computing the coefficients of A(z/gamma)
    ai_perceptual = a_coefficients*(gamma**np.arange(len(a_coefficients)))

    # Extracting frame_shift samples from the LPC analysis frame
    # and passing them through A(z)/A(z/gamma)
    speech_frame = input_frame[int((frame_length-frame_shift)/2):int((frame_length-frame_shift)/2+frame_shift)]
    LP_residual, z_inv = signal.lfilter(a_coefficients, 1, speech_frame, zi=z_inv)
    perceptual_speech, z_gamma_s = signal.lfilter(1, ai_perceptual, LP_residual, zi=z_gamma_s)

    # Building the long-term prediction codebook and filtering it
    LTP_codebook = np.zeros((frame_shift, LTP_max_delay))
    for j in range(LTP_max_delay):
         LTP_codebook[:,j] = excitation_buffer[j:j+frame_shift]
    LTP_codebook_filt = signal.lfilter(1, ai_perceptual, LTP_codebook, axis=0)
        
    # Filtering the stochastic codebook (all column vectors)
    codebook_filt = signal.lfilter(1, ai_perceptual, codebook, axis=0)
    
    # Finding the best predictor in the LTP codebook
    ringing, x = signal.lfilter(1, ai_perceptual, np.zeros(frame_shift), zi=z_gamma_e)
    sig = perceptual_speech - ringing
    LTP_gain, LTP_index = fNbc.find_Nbest_components(sig, LTP_codebook_filt, 1)
    
    # Generating the corresponding prediction
    LT_prediction = np.dot(LTP_codebook[:,LTP_index],LTP_gain)
        
    # Finding speech_frame components in the filtered codebook
    # taking long term prediction into account 
    sig = sig - np.dot(LTP_codebook_filt[:,LTP_index],LTP_gain)
    gains, indices = fNbc.find_Nbest_components(sig,codebook_filt, N_components)
    
    # Generating the corresponding excitation as a weighted sum of
    # codebook vectors plus long-term prediction
    excitation = LT_prediction + np.dot(codebook[:,indices],gains)

    # Synthesizing CELP speech, and keeping track of the synthesis filter 
    # internal variables
    synt_frame, z_synt = signal.lfilter(1, a_coefficients, excitation, zi=z_synt)
    synt_speech_CELP_LTP_PF.extend(synt_frame)
   
    # Updating the internal variables of the percpetual filter applied to
    # the excitation
    ans, z_gamma_e = signal.lfilter(1, ai_perceptual, excitation,zi=z_gamma_e)

    # Updating the excitation buffer for long-term prediction
    excitation_buffer[:LTP_max_delay]=excitation_buffer[frame_shift:LTP_max_delay+frame_shift]
    excitation_buffer[LTP_max_delay:LTP_max_delay+frame_shift]=excitation

    # Screen output
    if (i + 1) % 10 == 0:
        print(f'frame {i + 1:3d}')

    if i==134:
        plt.figure(figsize=(10, 6))

        W, H  = signal.freqz([1], a_coefficients, worN=512)
        W, HS = signal.freqz(input_frame, [1], worN=512)
        W, HP = signal.freqz(a_coefficients, ai_perceptual, worN=512)
        W, HE = signal.freqz(speech_frame - synt_frame, [1], worN=512)
        H_dB  = 20*np.log10(np.abs(H))
        HS_dB = 20*np.log10(np.abs(HS))
        HP_dB = 20*np.log10(np.abs(HP))
        HE_dB = 20*np.log10(np.abs(HE))
        
        plt.plot(W, HS_dB, '-', label='Input frame')
        plt.plot(W, H_dB, '--', label='Synthesis filter')
        plt.plot(W, HP_dB, '-.', label='Perceptual filter')
        plt.plot(W, HE_dB, ':', label='CELP residual')
        plt.xlabel('Frequency (rad/sample)')
        plt.ylabel('Magnitude (dB)')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.title('Spectral Envelope and Error Filtering (Frame 135)')

        plt.tight_layout()

synt_speech_CELP_LTP_PF = np.array(synt_speech_CELP_LTP_PF)

silence = np.zeros(int(2000), dtype=synt_speech_CELP_LTP_PF.dtype)
synt_speech_CELP_LTP_PF_padded = np.concatenate((synt_speech_CELP_LTP_PF, silence))
sd.play(synt_speech_CELP_LTP_PF_padded,8000)
sd.wait()

# %%
'''
While using less stochastic components as in the previous example,
synthetic speech quality is maintained. 

One can roughly estimate the corresponding bit-rate. Assuming 30 bits are enough 
for the prediction coefficients and each gain factor is quantized on 5
bits, we have to send for each frame: 30 bits [ai] + 7 bits [LTP index] 
+ 5 bits [LTP gain] + 2 [stochastic components] *(9 bits [index]
+ 5 bits [gain]) = 70 bits every 5 ms, i.e. 14 kbits/s.
The so-called "enhanced full rate" codec of GSMs implements a particular
version of CELP, termed as Algebraic CELP (ACELP) in which codebook
samples can only take 0, +1, or -1 values. 
The bit rate is maintained as low as 8 kbits/s by sending prediction
coefficients only once every four frame.
'''

plt.figure  (figsize=(10,8))
plt.plot    (synt_speech_CELP_LTP_PF)
plt.title   ('Synthesized Speech with CELP (N=2)')
plt.xlabel  ('Time (samples)')
plt.ylabel  ('Amplitude')
plt.grid    (True)

# %%
vt.plot_spectrogram(synt_speech_CELP_LTP_PF)
# %%
'''
Appendix 1: MPE as a particular case of CELP
It is easy to change the CELP script we have given above to make it
simulate Multi-Pulse Excited (MPE) linear prediction, in which 
excitation is obtained by adjusting the position and amplitudes of a
limited number of impulses per frame, so as to minimize a perceptually
weighted error. Long-term prediction is also applied.
The only thing we need to change is the codebook, which we set as an
identity matrix (each pulse being an excitation component).
In this test, we use 5 pulses for every 5 ms synthetic frame.
'''
frame_length    = 240      # length of the LPC analysis frame
frame_shift     = 40       # length of the excitation and synthesis frames
codebook_size   = 40       # number of vectors in the codebook
N_components    = 5        # number of codebook components per frame
LTP_max_delay   = 256      # maximum long-term prediction delay (in samples)
gamma           = 0.8      # perceptual factor

sample_rate, audio_raw = wavfile.read("../../audio_samples/speech.wav")
audio = audio_raw/32768

# Initializing internal variables
z_inv=np.zeros(10)  # inverse filter
z_synt=np.zeros(10) # synthesis filter
z_gamma_s=np.zeros(10) # perceptual filter applied to speech
z_gamma_e=np.zeros(10) # perceptual filter applied to excitatoin
synt_speech_MPE = []
excitation_buffer=np.zeros(LTP_max_delay+frame_shift)

# Building the stochastic excitation codebook
codebook = np.eye(frame_shift,codebook_size)

for i in range(int((len(audio)-frame_length+frame_shift)/frame_shift)):
    
    input_frame = audio[i*frame_shift:i*frame_shift+frame_length]

    # LPC analysis of order 10
    windowed_frame = input_frame*np.hamming(frame_length)
    a_coefficients, sigma_squared = lpc_bfc.lpc_calculation(windowed_frame, 10)
    sigma = np.sqrt(sigma_squared)
        
    # Computing the coefficients of A(z/gamma)
    ai_perceptual = a_coefficients*(gamma**np.arange(len(a_coefficients)))

    # Extracting frame_shift samples from the LPC analysis frame
    # and passing them through A(z)/A(z/gamma)
    speech_frame = input_frame[int((frame_length-frame_shift)/2):int((frame_length-frame_shift)/2+frame_shift)]
    LP_residual, z_inv = signal.lfilter(a_coefficients, 1, speech_frame, zi=z_inv)
    perceptual_speech, z_gamma_s = signal.lfilter(1, ai_perceptual,LP_residual, zi=z_gamma_s)
    
    # Building the long-term prediction codebook and filtering it
    LTP_codebook = np.zeros((frame_shift,LTP_max_delay))
    for j in range(LTP_max_delay):
        LTP_codebook[:,j] = excitation_buffer[j:j+frame_shift]
    LTP_codebook_filt = signal.lfilter(1, ai_perceptual, LTP_codebook, axis=0)
        
    # Filtering the stochastic codebook (all column vectors)
    codebook_filt = signal.lfilter(1, ai_perceptual, codebook, axis=0)
    
    # Finding the best predictor in the LTP codebook
    ringing, x = signal.lfilter(1, ai_perceptual, np.zeros(frame_shift),zi=z_gamma_e)
    sig = perceptual_speech - ringing
    LTP_gain, LTP_index = fNbc.find_Nbest_components(sig,LTP_codebook_filt, 1)
    
    # Generating the corresponding prediction
    LT_prediction = np.dot(LTP_codebook[:,LTP_index],LTP_gain)

    # Finding speech_frame components in the filtered codebook
    # taking long term prediction into account 
    sig = sig - np.dot(LTP_codebook_filt[:,LTP_index],LTP_gain)
    gains, indices = fNbc.find_Nbest_components(sig,codebook_filt, N_components)
    
    # Generating the corresponding excitation as a weighted sum of
    # codebook vectors plus long-term prediction
    excitation = LT_prediction + np.dot(codebook[:,indices],gains)
        
    # Synthesizing CELP speech, and keeping track of the synthesis filter 
    # internal variables
    synt_frame, z_synt = signal.lfilter(1, a_coefficients, excitation, zi=z_synt)
    synt_speech_MPE.extend(synt_frame)
   
    # Updating the internal variables of the percpetual filter applied to
    # the excitation
    ans, z_gamma_e = signal.lfilter(1, ai_perceptual, excitation, zi=z_gamma_e)

    # Updating the excitation buffer for long-term prediction
    excitation_buffer[:LTP_max_delay]=excitation_buffer[frame_shift:LTP_max_delay+frame_shift]
    excitation_buffer[LTP_max_delay:LTP_max_delay+frame_shift]=excitation

    LP_residual, z_inv = signal.lfilter(a_coefficients, 1, speech_frame, zi=z_inv)
     
    # Screen output
    if (i + 1) % 10 == 0:
        print(f'frame {i + 1:3d}')

    if i==139:
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(LP_residual, label='LPC residual')
        plt.plot(excitation, '--', label='MPE excitation')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(speech_frame, label='original speech')
        plt.plot(synt_frame, '-.', label='synthetic speech')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()

synt_speech_MPE = np.array(synt_speech_MPE)

silence = np.zeros(int(2000), dtype=synt_speech_MPE.dtype)
synt_speech_MPE_padded = np.concatenate((synt_speech_MPE, silence))
sd.play(synt_speech_MPE_padded,8000)
sd.wait()

# %%
'''
One can again roughly estimate the corresponding bit-rate. 
Each frame requires : 30 bits [ai] + 7 bits
[LTP index] + 5 bits [LTP gain] + 5 [pulses] *(6 bits [position]
+ 5 bits [gain]) = 97 bits every 5 ms, i.e. 19.4 kbits/s.

The so-called "full rate" codec of GSMs implements a particular
version of MPE, termed as Regular Pulse Excited,(RPE) which runs with a
bit-rate of 13 kbits/s 
'''
