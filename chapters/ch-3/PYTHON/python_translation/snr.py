
import numpy as np
from scipy import signal

def snr(sig, signal_plus_noise, max_shift):
    """
    Returns the signal-to-noise ratio (in dB) computed from the input
    signals.
    sig: the reference (clean) signal.
    signal_plus_noise: the signal to compare against 'sig'. 
    max_shift: maximum time-shift (in samples) allowed between 'sig' and 'signal_plus_noise'. The actual shift is found from the peak of the cross-correlation between the two signals, and is taken into account when estimating the noise.
    If the two signals have different lengths, the shorter length is used.
    snr_value: the estimated SNR, in dB.

    T DUTOIT, 13:49 12/03/2007
    """

    sig_length = min(len(sig), len(signal_plus_noise))
    sig_length = (sig_length//2)*2 # make it even
    sig = sig[:sig_length]
    signal_plus_noise = signal_plus_noise[:sig_length]

    length = min(max(10*max_shift, 1000), sig_length-1)
    half_len = length//2
    center = sig_length//2
    noisy_part = signal_plus_noise[center - half_len : center + half_len]
    signal_part = sig[center - half_len : center + half_len]
    cross_correlation = signal.correlate(noisy_part, signal_part)
    zero_lag_idx = len(noisy_part) - 1
    print(zero_lag_idx)
    positive_lags = cross_correlation[zero_lag_idx : zero_lag_idx + max_shift + 1]
    
    shift = np.argmax(positive_lags)

    tmp = signal_plus_noise[shift:]
    noise = tmp - sig[:len(tmp)]

    var_signal_dB = 10 * np.log10(np.maximum(np.var(sig), 1e-20))
    var_noise_dB = 10 * np.log10(np.maximum(np.var(noise), 1e-20))
    snr_value = var_signal_dB - var_noise_dB

    return snr_value
