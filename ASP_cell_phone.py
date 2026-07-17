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

plot_periodogram(input_frame_for_letter_e)

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

def plot_filter_responses(ai):
    """
    Replicates MATLAB's freqz plots for the LPC filters.
    ai: The linear prediction coefficients array.
    """
    # Synthesis filter is: 1 / A(z) (all-poles filter)
    # The 'b' (numerator) coefficients are 1, the 'a' (denominator) is our ai array
    W, H = signal.freqz(1, ai, worN=512)
    
    # Inverse filter is: A(z)
    # The 'b' coefficients are now ai, the 'a' coefficients are now 1
    WI, HI = signal.freqz(ai, 1, worN=512)
    
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
    plt.show    ()

plot_filter_responses(a_coefficients)

# %%


