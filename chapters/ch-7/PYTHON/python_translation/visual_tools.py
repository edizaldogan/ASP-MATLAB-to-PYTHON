
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def plot_periodogram(frame, fs):

    f, Pxx = signal.periodogram(frame, 
                                    fs=fs, 
                                    window='boxcar', 
                                    detrend=False,
                                    nfft=512)
 
    f_normalized = 2*f/fs
    Pxx_db = 10*np.log10(np.maximum(Pxx, 1e-10))
    
    plt.figure  (figsize=(10, 8))
    plt.plot    (f_normalized, Pxx_db)
    plt.title   ('Periodogram Power Spectral Density Estimate')
    plt.ylabel  ('Power/frequency (dB/(rad/sample))')
    plt.xlabel  ('Normalized Frequency ($\\times \\pi$ rad/sample)')
    plt.xlim    (0, 1)
    plt.grid    (True)

def plot_spectrogram(audio, fs=8000):

    f, t, Sxx = signal.spectrogram(audio,
                                   fs=fs,
                                   window=np.hamming(40),
                                   nperseg=40,
                                   noverlap=20,
                                   nfft=512)
    plt.figure      (figsize=(10, 8))
    plt.pcolormesh  (t, f, 10 * np.log10(Sxx), shading='auto', cmap='jet')
    plt.title       ("Spectrogram")
    plt.ylabel      ('Frequency (Hz)')
    plt.xlabel      ('Time (s)')

def plot_welch(frame, fs=2*np.pi):
    L = len(frame)
    nperseg = int(L / 4.5)
    noverlap = nperseg // 2

    f, Pxx = signal.welch(frame, 
                          fs=fs, 
                          nfft=512,
                          window='hamming', 
                          nperseg=nperseg, 
                          noverlap=noverlap,
                          detrend=False)
    
    Pxx_db = 10 * np.log10(Pxx)
    plt.figure  (figsize=(10, 8))
    plt.plot    (f / np.pi, Pxx_db)
    plt.title   ('Welch Power Spectral Density Estimate')
    plt.xlabel  ('Normalized Frequency ($\\times \\pi$ rad/sample)')
    plt.ylabel  ('Power/frequency (dB/(rad/sample))')
    plt.xlim    (0, 1)
    plt.grid    (True)

def pwelch(x, fs=2):
    """MATLAB-like |pwelch(x,[],[],[],fs)|: Welch PSD estimate.

    Uses MATLAB's defaults (8 Hamming segments with 50% overlap,
    nfft = max(256, next power of 2 of the segment length)) and returns the
    one-sided frequency axis together with the PSD in dB. With fs=2 the
    frequency axis is normalized: f/(Fs/2), in [0, 1].
    """
    x = np.asarray(x, dtype=float).ravel()
    nperseg = max(8, int(len(x) / 4.5))
    noverlap = nperseg // 2
    nfft = max(256, int(2 ** np.ceil(np.log2(nperseg))))

    f, Pxx = signal.welch(x, fs=fs, window='hamming', nperseg=nperseg,
                          noverlap=noverlap, nfft=nfft, detrend=False)
    Pxx_db = 10 * np.log10(np.maximum(Pxx, 1e-30))
    return f, Pxx_db


def plot_pwelch(x, fs=2, new_figure=True, label=None, **kwargs):
    """Plots the |pwelch| PSD estimate of |x| (dB) versus normalized frequency."""
    f, Pxx_db = pwelch(x, fs=fs)
    if new_figure:
        plt.figure(figsize=(10, 8))
    plt.plot(f, Pxx_db, label=label, **kwargs)
    plt.ylabel('Power/frequency (dB)')
    plt.xlabel('Normalized Frequency : f/(Fs/2)')
    plt.grid(True)
    return f, Pxx_db


def plot_zplane(b, a):

    max_len = max(len(b), len(a))
    b_padded = np.pad(b, (0, max_len - len(b)), 'constant')
    a_padded = np.pad(a, (0, max_len - len(a)), 'constant')

    zeros = np.roots(b_padded)
    poles = np.roots(a_padded)
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    unit_circle = plt.Circle((0,0), 1, color='blue',fill=False, 
                             linestyle='--')
    ax.add_patch(unit_circle)
    plt.axvline(0, color='blue',linestyle='--')
    plt.axhline(0, color='blue',linestyle='--')

    plt.plot(np.real(zeros), np.imag(zeros), 'o', markersize=8, markerfacecolor='None',
            color='blue', label='Zeros')
    
    plt.plot(np.real(poles), np.imag(poles), 'x', markersize=8, 
            color='blue', label='Poles')

    def print_number_of_roots(roots):
        print(len(roots))
        if len(roots) == 0:
            return
        rounded_roots = np.round(roots, 3)
        unique_roots, counts = np.unique(rounded_roots, return_counts=True)
        
        for root, count in zip(unique_roots, counts):
            if count > 1:
                plt.text(np.real(root) + 0.02, np.imag(root) + 0.02, str(count), fontsize=20)

    print_number_of_roots(zeros)
    print_number_of_roots(poles)
        
    plt.title('Pole-Zero Plot (Z-Plane)')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.axis('equal') 
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.legend()
