
import numpy as np
from tools import levinson


def shaping_filter_design(desired_frequency_response_dB, N_coef):
    '''
    [b0, ai] = shaping_filter_design(desired_frequency_response_dB, N_coef)
    computes the coefficients of the auto-regressive filter:

                           b0
    G(z) = ---------------------------------------------
                         -1                    -N_coef+1
           ai[0] + ai[1]z   + ... + ai[N_coef]z

    with ai[0] = 1, from the modulus of its desired_frequency_response
    (in dB) and the order N_coef.

    The coefficients are obtained as follows: if zero-mean and unity variance
    noise is provided at the input of the filter, the PSD of its output is
    given by desired_frequency_response. Setting the coefficients so that
    this PSD best matches desired_frequency_response is thus obtained by
    applying the Levinson algorithm to the autocorrelation coefficients of
    the output signal (computed itself from the IFFT of the
    desired_frequency_response).
    '''
    desired_frequency_response_dB = np.asarray(
        desired_frequency_response_dB, dtype=float).ravel()
    N = 2 * len(desired_frequency_response_dB)

    # Autocorrelation coefficients
    filter_response = 10.0 ** (desired_frequency_response_dB / 10)
    full_response = np.concatenate((filter_response,
                                    [filter_response[N // 2 - 1]],
                                    filter_response[N // 2 - 1:0:-1]))
    rk = np.real(np.fft.ifft(full_response))

    # Levinson's procedure
    ai, error = levinson(rk[0:N_coef + 1], N_coef)
    b0 = np.sqrt(error)

    return b0, ai
