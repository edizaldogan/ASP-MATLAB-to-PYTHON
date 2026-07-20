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

import matplotlib.pyplot as plt
# Set global figure parameter 
# This makes all the background colors of figures white by default.
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

def plot_periodogram(frame, fs):
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

    # detrend=False mimic MATLAB's normalized PSD assumption
    f, Pxx = signal.periodogram(frame, 
                                    fs=fs, 
                                    window='boxcar', 
                                    detrend=False,
                                    nfft=512)
    
    # Map the frequency axis between 0 and 1 (as multiples of pi)
    # f_max = fs/2
    # f_normalized = f/f_max = f/(fs/2) = 2*f/fs
    f_normalized = 2*f/fs
    
    # Convert the linear PSD to Decibels (dB)
    Pxx_db = 10 * np.log10(Pxx)
    
    plt.figure  (figsize=(10, 8))
    plt.plot    (f_normalized, Pxx_db)
    plt.title   ('Periodogram Power Spectral Density Estimate')
    plt.ylabel  ('Power/frequency (dB/(rad/sample))')
    plt.xlabel  ('Normalized Frequency ($\\times \\pi$ rad/sample)')
    plt.xlim    (0, 1)
    plt.grid    (True)

plot_periodogram(input_frame_for_letter_e, 2*np.pi)

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

def autocorrelation_calculation(input_frame, order):
    '''
    Takes in the 240 element frame and calculates the autocorrelation
    values R[0], R[1],..., R[p] where p is the order.
    '''
    # Autocorrelation vector will be in the form: r=[R[0] R[1] R[2] ... R[p]]
    autocorrelation_vector = []
    # p (the order) goes from 0 to p.
    for p in range(order + 1):
        sum = 0
        # n goes from 0 to 240-p.
        for n in range(len(input_frame)-p):
            # By definition, we multiply the signal itself by its 
            # k-unit shifted version and add the results.
            sum = sum + input_frame[n]*input_frame[n+p]
        autocorrelation_vector.append(float(sum))
    return autocorrelation_vector

def lpc_calculation(input_frame_for_letter_e, order):
    '''
    For a given frame we try to find the optimal ai coefficients that 
    minimizes the expectation of the residual energy argmin(E[e^2[n]]).
    This function solves the Yule-Walker equations by brute-force.
    Yule-Walker equations are in the following matrix format: R*a=r.
    '''
    
    # Create the autocorrelation vector containing R[0] to R[p]
    autocorrelation_vector = autocorrelation_calculation(input_frame_for_letter_e, order)

    # CHECK POINT - autocorrelation vector
    counter = 0
    print('autocorrelation vector is: ')
    for element in autocorrelation_vector:
        print('R[',counter,'] = ', element, sep='')
        counter += 1
    print('')
    
    R = [] # pxp matrix
    # start from row 1 to row p
    # Remember: Autocorrelation vector = [R[0] R[1] R[2] ... R[p]]
    for i in range(1,order+1):
        R_row_i = []
        for j in range(1,order+1):
            # Row i=1 of R is R[0], R[-1](or R[1]), ..., R[1-p] (or R[p-1])
            # Row i=2 of R is R[1], R[0], R[-1](or R[1]), ..., R[2-p] (or R[p-2])
            # ...
            # Row i=p of R is R[p-1], R[p-2], R[p-3], ..., R[0]
            # It is enough to check the absolute value of the difference i and j
            # because R[k]=R[-k] is satisfied for any k.
            R_row_i.append(autocorrelation_vector[abs(i-j)])
        R.append(R_row_i)
    
    # CHECK POINT - autocorrelation matrix R
    row_counter = 0
    print('autocorrelation matrix (R) is: ')
    for row in R:
        col_counter = 0
        for element in row:
            value = abs(row_counter-col_counter)
            print(f'R[{row_counter}][{col_counter}]=R[{value}]={element:>6.3f}', end=' | ')
            col_counter += 1
        print('')
        row_counter += 1
    print('')

    # Yule-Walker Equations expanded form: 
    # a1*R[k-1] + a2*R[k-2] +...+ ap*R[k-p] = -R[k] where k=1,2,...,order
    # In matrix form, right hand side is: -[R[1] R[2] ... R[p]]

    # 11 elements original vector, we will use it later
    a_v_o = autocorrelation_vector.copy()

    # To obtain the right hand side, we need to negate the autocorrelation vector:
    for i in range(len(autocorrelation_vector)):
        autocorrelation_vector[i] = -autocorrelation_vector[i]
    
    # CHECK POINT - Negated autocorrelation vector
    counter = 0
    print('Negated autocorrelation vector is: ')
    for element in autocorrelation_vector:
        print('R[',counter,'] = ', element, sep='')
        counter += 1
    print('')

    # Exclude the first term R[0]
    autocorrelation_vector.pop(0) # R[0] is taken out
    # Call this new autocorrelation_vector as a_v_rhs
    a_v_rhs = autocorrelation_vector # px1 vector
    # a_v_rhs = -[R[1] R[2] ... R[p]]

    # CHECK POINT - Right hand side vector
    counter = 1
    print('Right hand side vector (a_v_rhs) is: ')
    for element in a_v_rhs:
        print('R[',counter,'] = ', element, sep='')
        counter += 1
    print('')

    '''
    Now that R(autocorrelation matrix) and a_v_rhs(autocorrelation vector) are 
    formed, we start eliminating the unknowns by replacing them in terms of the 
    rest of the unknowns using Gaussian elimination.   
    '''

    # Gaussian Elimination Algorithm
    for i in range(order): # picking the i_th element of i_th row
        coef_of_the_a_to_be_eliminated = R[i][i]
        for j in range(i+1,order): # finding multipliers under row i
            multiplier = R[j][i] / coef_of_the_a_to_be_eliminated
            for k in range(i, order): # executing subtraction of rows
                R[j][k] = R[j][k] - (multiplier * R[i][k])
            a_v_rhs[j] = a_v_rhs[j] - (multiplier * a_v_rhs[i])

    # Obtaining a_coefficients = [a1 a2 ... ap]:
    a_coefficients = [0]*order # px1 lpc coefficients vector
    
    # Begin from the right
    for i in range(order-1, -1, -1):
        current_sum = 0
        # Notice that below will be skipped in the first loop
        for j in range(i + 1, order): # only right side concerns us.
            current_sum = current_sum + (R[i][j] * a_coefficients[j])
            
        # Rearrange the Yule-Walker Equation, leave lpc coef. alone.
        a_coefficients[i] = (a_v_rhs[i] - current_sum) / R[i][i]
        
    # By definition of LPC, the first coefficient a0 is always 1.
    # We insert 1 at the very beginning of our array.
    a_coefficients.insert(0, 1) # (p+1)x1 vector
    # Obtaining a_coefficients = [a0 a1 a2 ... ap]:

    # Residual energy: E = a0*R[0] + a1*R[1] + a2*R[2] + ... + ap*R[p]
    sigma_squared = 0
    for i in range(order+1):
        sigma_squared = sigma_squared + (a_coefficients[i] * a_v_o[i])
    # MATLAB always divide the summation by the length of the window:
    sigma_squared = sigma_squared/len(input_frame_for_letter_e)

    return a_coefficients, sigma_squared

# Notice that the input frame has 240 elements in it.
a_coefficients, sigma_squared = lpc_calculation(input_frame_for_letter_e, 10)
sigma = np.sqrt(sigma_squared)

# CHECK POINT - lpc coefficients, sigma_squared, sigma
counter = 0
print('lpc coefficients are: ')
for element in a_coefficients:
    print('a_',counter,' = ', element, sep='')
    counter += 1
print('')
print('sigma_squared = ', sigma_squared)
print('sigma = ', sigma)
print('')

# %%

'''
The estimation algorithm inside LPC is called the Levinson-Durbin
algorithm. It chooses the coefficients of an FIR filter A(z) so that when
passing the input frame into A(Z), the output, termed as the prediction
residual, has minimum energy. It can be shown that this leads to a filter
which has anti-resonances wherever the input frame has a formant. For
this reason, the A(z) filter is termed as the "inverse" filter. Let us
plot its frequency response (on 512 points), and superimpose it to that
of the "synthesis" filter 1/A(z).
'''

# Synthesis filter is: 1 / A(z) (all-poles filter)
# The 'b' (numerator) coefficients are 1, the 'a' (denominator) is our ai array
W, H = signal.freqz(1, a_coefficients, worN=512)

# Inverse filter is: A(z)
# The 'b' coefficients are now ai, the 'a' coefficients are now 1
WI, HI = signal.freqz(a_coefficients, 1, worN=512)

plt.figure(figsize=(10, 8))

x_tick_marks = np.arange(0, 1.1, 0.1)
# W/np.pi normalizes the X-axis to the range 0 to 1
plt.plot    (W/np.pi, 20*np.log10(np.abs(H)),'-', label='Synthesis filter 1/A(z)')
plt.plot    (WI/np.pi, 20*np.log10(np.abs(HI)),'--', label='Inverse filter A(z)')
plt.xlabel  ('Normalized frequency ($\\times \\pi$ rad/sample)')
plt.ylabel  ('Magnitude (dB)')
plt.xticks  (x_tick_marks)
plt.legend  ()
plt.grid    (True)

# %%

'''
In other words, the frequency response of the filter 1/A(z) matches the
spectral amplitude envelope of the frame. Let us superimpose this
frequency response to periodogram of the vowel.

NB: the |periodogram| function of MATLAB actually shows the so-called
one-sided periodogram, which has twice the value of the two-sided
periodogram in [0,Fs/2]. In order to force MATLAB to show the real value
of the two-sided periodogram in [0, Fs/2], we claim Fs=2.
'''

plot_periodogram(input_frame_for_letter_e, 2) # fs=2
plt.plot(W/np.pi,20*np.log10(sigma*abs(H)));
plt.show()

'''
PHYSICAL AND MATHEMATICAL INTERPRETATION OF THE PLOTS:
1. The Periodogram (Blue Line - The Source / Vocal Cords):
   This plot is the brute-force Fast Fourier Transform (FFT) of the raw 240-sample 
   audio frame of letter 'e'. Because it is computed from many independent data points,
   it has the capacity to produce highly detailed (spiky), rapid oscillations. These 
   sharp and dense spikes (harmonics) represent the fundamental frequency and the raw 
   vibration of the vocal cords.

2. The 1/A(z) LPC Envelope (Orange Line - The Filter / Vocal Tract):
   This plot represents the frequency response of our synthesis filter, 1/A(z).
   When we call `signal.freqz(1, ai, worN=512)`, the algorithm essentially takes 
   our small 11-element coefficient array [1, a1, a2, ..., a10], zero-pads it up 
   to 512 points [1, a1, a2, ..., a10, 0, ..., 0], and computes its FFT (which 
   gives A(z)). Taking the inverse of this result yields the 1/A(z) envelope.

3. Why is the envelope so smooth while the periodogram is oscillatory?
   Since the orange line is generated from only 11 non-zero values (the LPC coefficients), 
   it physically lacks the information capacity to create sharp, rapid spikes. It can 
   only produce broad, smooth oscillations. 
   
Conclusion:
By solving the Yule-Walker equations, we optimized these 11 numbers so that their 
smooth FFT inverse tightly wraps around the highest energy peaks of the raw periodogram. 
This perfectly isolates the macroscopic shape of the vocal tract (the formants) while 
filtering out the microscopic vocal cord vibrations.
'''

# %%

'''
Formants appear again, much better than before. As a matter of fact, LP
modelling acts as a parametric spectral estimator, resulting in a smooth
estimate.

In other words, the LPC fit has automatically adjusted the poles of the
synthesis filter close to the unit circle at angular positions chosen to
imitate formant resonances.
'''

# a_coefficients = [1 a1 a2 ... ap] form A(z) and 1/A(z) as follows:
# A(z) = 1 + a1*z^-1 + a2*z^-2 + ... + ap*z^-p
# 1/A(z) = 1/(1 + a1*z^-1 + a2*z^-2 + ... + ap*z^-p)

def plot_zplane(b, a):
    """
    Replicates MATLAB's zplane(b, a) functionality.
    b: numerator coefficients
    a: denominator coefficients
    """
    
    # first equate the lengths by padding zeros, we shouldn'T miss the zeros.
    max_len = max(len(b), len(a))
    b_padded = np.pad(b, (0, max_len - len(b)), 'constant')
    a_padded = np.pad(a, (0, max_len - len(a)), 'constant')
    # Then find the roots:
    # Notice there will be zeros on the origin because b=[1, 0, ..., 0]
    zeros = np.roots(b_padded)
    poles = np.roots(a_padded)
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    unit_circle = plt.Circle((0,0), 1, color='blue',fill=False, linestyle='--')
    ax.add_patch(unit_circle)
    plt.axvline(0, color='blue',linestyle='--')
    plt.axhline(0, color='blue',linestyle='--')

    plt.plot(np.real(zeros), np.imag(zeros), 'o', markersize=8, markerfacecolor='None',
            color='blue', label='Zeros')
    
    plt.plot(np.real(poles), np.imag(poles), 'x', markersize=8, 
            color='blue', label='Poles')
        
    plt.title('Pole-Zero Plot (Z-Plane)')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.axis('equal') 
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.grid(True)
    plt.legend()

plot_zplane([1], a_coefficients)

'''
Are the poles inside the unit circle by coincidence?
Absolutely not. This is a fundamental mathematical guarantee of the Yule-Walker 
(Autocorrelation) method. For the synthesis filter 1/A(z) to be BIBO stable, all 
its poles must lie strictly inside the unit circle (|z| < 1). The positive-definite 
Toeplitz structure of the Yule-Walker equations mathematically ensures that the 
roots of the resulting A(z) polynomial will always satisfy this stability criterion. 
Otherwise, the synthesized speech would exponentially diverge to infinity.
'''

# %%

'''
If we apply the inverse of this filter to the input frame, we obtain the
prediction residual.
'''

def plot_filter(b,a,x):
    # b = [b0 b1 ... bn]
    # a = [a0 a1 ... am]
    # x = input frame
    # Y(z) = (b0+b1*z^-1+...+bn*z^-n)*X(z)/(a0+a1*z^-1+...+bn*z^-m)
    # e[n] = x[n]+b1*x[n-1]+b2*x[n-2]+...+bp*x[n-p] if a=[1]
    filter_output = signal.lfilter(b, a, x)
    plt.figure(figsize=(10, 8))
    plt.plot(filter_output, color='blue')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    return filter_output

LP_residual = plot_filter(a_coefficients,[1],input_frame_for_letter_e)

# %%

'''
Let us compare the spectrum of this residual to the original spectrum.
The new spectrum is approximately flat; its fine spectral details,
however, are the same as those of the analysis frame. In particular, its
pitch and harmonics are preserved. 
'''

plot_periodogram(LP_residual, 2*np.pi)

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

excitation = np.zeros(240) # 240 element frame with 65 pitch period
excitation[::65] = 1 # we have a one in every 65 index (0, 65, 130...)
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

plot_periodogram(gain*excitation, 2*np.pi)

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

plot_periodogram(synt_frame, 2*np.pi)

# %%

'''
3. Linear prediction synthesis of 30 ms of unvoiced speech
It is easy to apply the same process to an unvoiced frame, and compare
the final spectra again. Let us first extract an unvoiced frame and plot
it.
'''

# The following interval corresponds to the time interval of letter 'c'
# from the word 'circuits'.
input_frame_for_letter_c    = audio[4499:4739]
time                        = np.arange(0,240,1)
plt.figure  (figsize=(10, 8))
plt.plot    (input_frame_for_letter_c)
plt.title   ('Zooming on the letter "c"')
plt.xlabel  ('Time (samples)')
plt.ylabel  ('Amplitude')
plt.grid    (True)

# %%

'''
As expected, no clear periodicity appears.

Now let us see the spectral content of this speech frame. Notice that,
since we are dealing with noisy signals, we use the averaged periodogram
to have a better estimate power spectral densities, although with less
frequency resolution than using a simple periodogram. The MATLAB |pwlech|
function does  this, with 8 sub-frames by default and 50% overlap.
'''

def plot_welch(frame, fs=2*np.pi):
    """
    Replicates MATLAB's default pwelch(input_frame) behavior.
    Divides the signal into 8 sub-frames with 50% overlap using a Hamming window.
    Plots the Averaged Periodogram with normalized frequency.
    """
    
    # 8 windows: 0-S, 0.5S-1.5S, ..., 3.5S-4.5S (53 samples each)
    L = len(frame)
    nperseg = int(L / 4.5)
    noverlap = nperseg // 2
    
    # PSD calculation
    f, Pxx = signal.welch(frame, 
                          fs=fs, 
                          nfft=512,
                          window='hamming', 
                          nperseg=nperseg, 
                          noverlap=noverlap,
                          detrend=False)
    
    # dB conversion
    Pxx_db = 10 * np.log10(Pxx)
    
    plt.figure  (figsize=(10, 8))
    plt.plot    (f / np.pi, Pxx_db)
    plt.title   ('Welch Power Spectral Density Estimate')
    plt.xlabel  ('Normalized Frequency ($\\times \\pi$ rad/sample)')
    plt.ylabel  ('Power/frequency (dB/(rad/sample))')
    plt.xlim    (0, 1)
    plt.grid    (True)

plot_welch(input_frame_for_letter_c)

# %%

'''
Let us now apply an LP model of order 10, and synthesize a new frame.
Synthesis is performed by all-pole filtering a Gaussian white noise frame
with standard deviation set to the prediction residual standard
deviation, sigma.
'''

a_coefficients, sigma_squared = lpc_calculation(input_frame_for_letter_c, 10)
sigma = np.sqrt(sigma_squared)
# CHECK POINT - lpc coefficients, sigma_squared, sigma
counter = 0
print('lpc coefficients are: ')
for element in a_coefficients:
    print('a_',counter,' = ', element, sep='')
    counter += 1
print('')
print('sigma_squared = ', sigma_squared)
print('sigma = ', sigma)
print('')

excitation = np.random.randn(240) # Gaussian white noise
# excitation is passed throuhg the the filter that models the vocal tracts
synt_frame = plot_filter(sigma,a_coefficients,excitation)
plt.show()

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

plot_welch(synt_frame)

# %%

