
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

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