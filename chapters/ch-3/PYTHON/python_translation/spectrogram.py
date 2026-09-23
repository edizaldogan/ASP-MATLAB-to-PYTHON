
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

def plot_spectrogram(audio, nfft, fs, window):
    """
    Plots the spectrogram of 'audio', sampled at 'fs' Hz, using an 'nfft'-point
    FFT and a Hanning window of length 'window'.
    """
    f, t, Sxx = signal.spectrogram(audio, 
                                   fs=fs,
                                   window=np.hanning(window),
                                   nperseg=window,
                                   noverlap=window/2,
                                   nfft=nfft,
                                   detrend=False,
                                   mode='magnitude')
    plt.figure      (figsize=(10, 8))
    plt.pcolormesh  (t, f, 20*np.log10(Sxx), shading='auto', cmap='jet')
    plt.title       ("Spectrogram")
    plt.ylabel      ('Frequency (Hz)')
    plt.xlabel      ('Time (s)')
