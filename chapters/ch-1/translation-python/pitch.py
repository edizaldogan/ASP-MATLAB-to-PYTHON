
import numpy as np
from scipy import signal
import lpc_brute_force_calculation as lpc_bfc

def pitch(frame):
    '''
    Estimates the fundamental period (in samples) of a 30 ms speech frame.
    Returns T0 = 0 if the frame is detected as unvoiced.
    T0 is computed from the maximum of the autocorrelation of the LPC residual.
    '''
    a_coeffs, sigma_square = lpc_bfc.lpc_calculation(frame, 10)
    lpc_residual = signal.lfilter(a_coeffs, [1], frame) # glottal sound 240 sample
    # autocorrelation of the residual (2N-1: 2*240-1 = 479 elements)
    C = np.correlate(lpc_residual, lpc_residual, mode='full')
    center = len(lpc_residual) - 1 # center index 239
    C_half = C[center : center + 134] # 134 elements (239-373)
    Cxx = C_half / C_half[0] # normalization # 134 elements
    # Fisrt 26 index are forcefully zeroed. Vocal cords cannot vibrate that fast.
    # 26 sample period equals 307HZ which is the upper limit for vocal cords.
    Cxx[0:26] = 0 # we do not pick the lag 0 case!
    # the autocorrelation matrix of course has the highest correlation at lag 0 
    # (first element) but we are searching for the pitch, which means the second 
    # highest peak in this matrix.
    Amax = np.max(Cxx)  # max value
    Imax = np.argmax(Cxx) # index of the max value

    # U/UV decision
    if Amax > 0.20: # vowel threshold
        T0 = Imax
    else:
        T0 = 0
    return T0
