
import numpy as np

# ----------------------------------------------------------------------
# xcorr
# ----------------------------------------------------------------------
def xcorr(x, maxlag=None, scale='none'):

    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    if maxlag is None:
        maxlag = N - 1
    maxlag = int(maxlag)

    # Full correlation through the FFT (fast for the frame sizes used here)
    n_fft = int(2 ** np.ceil(np.log2(2 * N - 1)))
    X = np.fft.fft(x, n_fft)
    c = np.real(np.fft.ifft(X * np.conj(X)))
    # c[0] is lag 0, c[-k] is lag -k
    c = np.concatenate((c[-maxlag:], c[:maxlag + 1])) if maxlag > 0 else c[:1]

    lags = np.arange(-maxlag, maxlag + 1)
    if scale == 'biased':
        c = c / N
    elif scale == 'unbiased':
        c = c / (N - np.abs(lags))
    elif scale == 'coeff':
        c = c / c[maxlag]
    return c

# ----------------------------------------------------------------------
# levinson
# ----------------------------------------------------------------------
def levinson(r, order):
    r = np.asarray(r, dtype=float).ravel()
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    if e <= 0:
        return a, max(e, 0.0)

    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k = -acc / e
        a_prev = a[1:i].copy()
        a[1:i] = a_prev + k * a_prev[::-1]
        a[i] = k
        e *= (1.0 - k * k)
        if e <= 0:  # numerically non positive-definite autocorrelation
            e = np.finfo(float).eps
            break
    return a, e

def uencode(u, n, v, signed=True):

    u = np.asarray(u, dtype=float)
    if v == 0:
        return np.zeros_like(u)
    q = 2.0 * v / 2 ** n
    u = np.clip(u, -v, v)
    codes = np.floor((u + v) / q)
    codes = np.minimum(codes, 2 ** n - 1)
    if signed:
        codes = codes - 2 ** (n - 1)
    return codes

def udecode(codes, n, v):
    if v == 0:
        return np.zeros_like(np.asarray(codes, dtype=float))
    q = 2.0 * v / 2 ** n
    return np.asarray(codes, dtype=float) * q

def soundsc(x, Fs=44100, blocking=False):

    x = np.asarray(x, dtype=float).ravel()
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    try:
        import sounddevice as sd
        sd.play(x, Fs, blocking=blocking)
    except Exception as err:                     # no PortAudio, no device, ...
        print('Audio playback unavailable (%s)' % err)

def message_to_bits(message):
    bits = np.array([list(format(ord(c), '07b')) for c in message], dtype=int)
    return bits.flatten(order='F')

def bits_to_message(bits):
    bits = np.asarray(bits, dtype=int).ravel()
    n_chars = len(bits) // 7
    chars = np.reshape(bits[:7 * n_chars], (n_chars, 7), order='F')
    weights = 2 ** np.arange(6, -1, -1)
    return ''.join(chr(int(c)) for c in chars.dot(weights))
