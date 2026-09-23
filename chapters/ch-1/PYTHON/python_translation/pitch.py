
import numpy as np
from scipy import signal
import lpc_brute_force_calculation as lpc_bfc

def pitch(frame):
    '''
    This function estimates the fundamental period (in samples) of a 30 ms
    speech frame sampled at 8 kHz. T0=0 if the frame is detexted as unvoiced.
    T0 is computed from the max. of the autocorrelation of the LPC residual.
    Voiced/unvoiced decision is based on the the ratio of this maximum by the
    variance of the residual. Pitch is searched for in the range
    [60Hz,300Hz], i.e. for autocorrelation indices in [26,133].
    '''
    a_coeffs, sigma_square = lpc_bfc.lpc_calculation(frame, 10)
    lpc_residual = signal.lfilter(a_coeffs, [1], frame)
    C = np.correlate(lpc_residual, lpc_residual, mode='full') # autocorrelation
    center = len(lpc_residual)-1
    C_half = C[center:center+134]
    Cxx = C_half/C_half[0] # relative to its variance
    Cxx[0:26] = 0
    Amax = np.max(Cxx)  # max value
    Imax = np.argmax(Cxx) # position of the maximum
    if Amax > 0.20: # *very* rough V/UV condition.
        T0 = Imax
    else:
        T0 = 0
    return T0
