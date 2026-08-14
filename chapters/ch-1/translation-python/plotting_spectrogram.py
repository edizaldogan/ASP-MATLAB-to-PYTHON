
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

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