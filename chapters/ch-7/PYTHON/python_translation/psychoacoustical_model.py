
import numpy as np
from scipy import signal


def psychoacoustical_model(audio_signal):
    '''
    masking_threshold = psychoacoustical_model( audio_signal ) returns
    the masking threshold deduced from a psychoacoustical analysis of
    the audio vector.

    This implementation is derived from the psycho-acoustic model #1 used in
    MPEG-1 Audio (see ISO/CEI norm 11172-3:1993 (F), pp. 122-128 or the
    MPEG1_psycho_acoustic_model1 function from Chapter 3). It is based on
    the same principles as those used in the MPEG model, but it is further
    adapted here so as to make it robust to additive noise (which is a
    specific constraint of watermarking and is not found in MPEG).
    '''
    audio_signal = np.asarray(audio_signal, dtype=float).ravel()
    N = len(audio_signal)

    # Power Spectral Density (PSD) computation
    window = np.hanning(N + 2)[1:-1]
    audio_fft = np.fft.fft(audio_signal * window)
    audio_PSD = 10 * np.log10(np.abs(audio_fft[0:N // 2]) ** 2 / N +
                              np.finfo(float).tiny)

    # Dynamic compression of the PSD over 4 "frequency critical bands"
    Nk = N // 8
    compressed_PSD = np.zeros(N // 2)
    for k in range(4):
        band = audio_PSD[k * Nk:k * Nk + Nk]
        mk = np.mean(band)
        compressed_PSD[k * Nk:k * Nk + Nk] = (band - mk) / 2 + mk

    # Spreading function application
    P = 10
    spreading_function = np.array([-0.0052, -0.008, 0.0134, 0.1057, 0.2405,
                                   0.3072, 0.2405, 0.1057, 0.0134, -0.008,
                                   -0.0052])

    input_PSD = np.concatenate((np.ones(P // 2) * compressed_PSD[0],
                                compressed_PSD,
                                np.ones(P // 2) * compressed_PSD[N // 2 - 1]))
    filtered_PSD = signal.lfilter(spreading_function, 1, input_PSD)
    filtered_PSD = filtered_PSD[P:P + N // 2]

    # Smoothing over the 3 "high frequency critical bands"
    masking_threshold = filtered_PSD.copy()

    # 2d critical band
    for i in range(Nk, 2 * Nk, 2):
        masking_threshold[i:i + 2] = np.mean(filtered_PSD[i:i + 2])

    # 3d/4th critical band
    for i in range(2 * Nk, 4 * Nk - 2, 3):
        masking_threshold[i:i + 3] = np.mean(filtered_PSD[i:i + 3])
    masking_threshold[N // 2 - 2:N // 2] = masking_threshold[N // 2 - 3]

    # Tuning : this value can be changed, so as to tune the inaudibility
    # constraint
    masking_threshold = masking_threshold - 5

    return masking_threshold
