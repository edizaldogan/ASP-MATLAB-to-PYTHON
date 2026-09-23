
import numpy as np

from PQMF32_prototype import PQMF32_prototype
from MPEG1_psycho_acoustic_model1 import MPEG1_psycho_acoustic_model1
from MPEG1_bit_allocation import MPEG1_bit_allocation
from tools import uencode, udecode


def mp3_codec(input_signal, Fs, bit_rate=192000, verbose=True):
    '''
    output_signal = mp3_codec( input_signal, Fs ) returns the signal
    resulting from an MPEG-1 Layer I coding/decoding operation applied to
    input_signal, sampled at Fs.
    '''
    input_signal = np.asarray(input_signal, dtype=float).ravel()

    # Pseudo QMF filters bank
    hn = PQMF32_prototype()
    PQMF32_Gfilters = np.zeros((32, 512))
    for i in range(32):
        t2 = ((2 * i + 1) * np.pi / (2 * 32)) * (np.arange(512) + 16)
        PQMF32_Gfilters[i, :] = hn * np.cos(t2)

    n_frames = int(np.floor((len(input_signal) - 512 + 32) / 32))
    if n_frames < 12:
        raise ValueError('Input signal is too short (at least 864 samples '
                         'are needed).')

    # Analysis: subbands[i] = PQMF32_Gfilters * input_signal[i*32 : i*32+512]
    # (done here with a single matrix product, for speed)
    frames = np.lib.stride_tricks.sliding_window_view(
        input_signal, 512)[0:n_frames * 32:32]
    if verbose:
        print('Analyzing %5d frames' % n_frames)
    subbands = frames @ PQMF32_Gfilters.T

    n_frames = (n_frames // 12) * 12
    quantized_subbands = np.zeros((n_frames, 32))
    for k in range(0, n_frames, 12):
        if verbose and k % 120 == 0:
            print('Processing frame %5d %5d' % (k + 1, n_frames))

        scale_factors = np.max(np.abs(subbands[k:k + 12, :]), axis=0)
        frame = input_signal[175 + k * 32:175 + k * 32 + 512]
        SMR = MPEG1_psycho_acoustic_model1(frame)[0]

        # Allocating bits for a target bit rate of 192 kbits/s
        N_bits = MPEG1_bit_allocation(SMR, bit_rate)[0]

        # Adaptive perceptual uniform quantization, using a mid-tread
        # quantizer in [-Max,+Max]
        for j in range(32):                      # for each sub-band
            if N_bits[j] != 0:
                n = int(N_bits[j])
                codes = uencode(subbands[k:k + 12, j], n,
                                scale_factors[j], signed=True)
                quantized_subbands[k:k + 12, j] = udecode(codes, n,
                                                          scale_factors[j])
            else:
                quantized_subbands[k:k + 12, j] = 0

    # Synthesis
    output_signal = np.zeros(len(input_signal))
    for i in range(n_frames):
        output_frame = PQMF32_Gfilters.T @ quantized_subbands[i, :]
        output_signal[i * 32:i * 32 + 512] += output_frame

    return output_signal
