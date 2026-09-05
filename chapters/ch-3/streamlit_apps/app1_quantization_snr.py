"""
Quantization & SNR Explorer
(number of quantization bits per sample can be manipulated)

Concept this teaches
---------------------
Every sub-band sample gets uniformly quantized with N bits before it is
sent. Fewer bits leads to smaller file, but noisier reconstruction. 
As you drag the bit-depth slider, you see the waveform get "steppy", the error signal
grow, and the SNR (in dB) drop in real time.

Run with:
    streamlit run app1_quantization_snr.py
"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quantization & SNR Explorer", layout="wide")
st.title("Quantization & SNR Explorer")
st.write(
    "This is exactly the mid-thread quantizer used in the MPEG-1 sub-band "
    "coder (Section 3.2.5). Drag the slider to change **only** the number "
    "of bits per sample, and watch the reconstructed signal and the SNR."
)

n_bits = st.slider("Number of bits per sample", min_value=1, max_value=16, value=4)

# test signal
Fs = 8000
duration = 0.02 # short so that we can see individual samples
t = np.arange(0, duration, 1 / Fs)
signal = np.sin(2*np.pi*220*t) + np.sin(2*np.pi*660*t)

def mid_thread_quantize(x, n_bits):
    """Same mid-thread quantizer as used throughout the chapter:
    alpha = 2^(n_bits-1); q = floor(alpha*x + 0.5) / alpha
    A mid-thread quantizer maps small values (including 0) to exactly 0,
    which avoids constant low-level "musical noise" on quiet signals.
    """
    alpha = 2**(n_bits-1)
    return np.floor(alpha*x+0.5)/alpha

quantized = mid_thread_quantize(signal, n_bits)
error = quantized - signal
var_signal = np.var(signal)
var_noise = max(np.var(error), 1e-20)
snr_db = 10*np.log10(var_signal / var_noise)

col1, col2 = st.columns([1, 3])
with col1:
    st.metric("Resulting SNR", f"{snr_db:.1f} dB")
    st.metric("Quantization levels", f"{2**n_bits}")
    st.caption(
        "Rule of thumb: each extra bit buys ~6 dB of SNR "
        "(this is the classic '6 dB per bit' rule)."
    )

with col2:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(t * 1000, signal, label="Original", linewidth=1.5)
    axes[0].step(t * 1000, quantized, where="mid", label="Quantized", color="tab:red")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Original vs. {n_bits}-bit quantized signal")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t * 1000, error, color="tab:green")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Error")
    axes[1].set_ylim(0.3,-0.3)
    axes[1].set_title("Quantization error (noise) — this is what gets added to the audio")
    axes[1].grid(alpha=0.3)

    st.pyplot(fig)

st.info(
    "In the real MPEG-1 coder, this same slider is set **independently for "
    "each of the 32 sub-bands**, by the bit-allocation algorithm, based on "
    "how much noise the psychoacoustic model says can be hidden in that "
    "band. Try dragging to 1-2 bits to hear/see how bad *uniform*, "
    "non-adaptive quantization would be everywhere at once."
)
