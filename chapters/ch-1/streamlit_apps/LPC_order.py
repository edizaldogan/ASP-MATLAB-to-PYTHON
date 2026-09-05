
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_toeplitz
from scipy.signal import freqz

st.set_page_config(page_title="Understanding LPC Order",
                   page_icon="🦈",layout="wide",
                   initial_sidebar_state="collapsed")
st.title('Understanding LPC Order')

lpc_order = st.slider("LPC Order (p)", min_value=2, max_value=60, value=10, step=1)

fs = 8000
frame_size = 240
t = np.arange(frame_size)/fs
# Formants at: 300Hz, 1400Hz, 2700Hz
example_signal = np.sin(2*np.pi*300*t) + 0.5*np.sin(2*np.pi*1400*t) + 0.8*np.sin(2*np.pi*2700*t)
frame = example_signal + np.random.randn(frame_size)/5 # add white noise
windowed_frame = frame*np.hamming(frame_size)

# Levinson Durbin Algorithm
from scipy.linalg import solve_toeplitz

def lpc_toeplitz(frame, order):
    r = [sum(frame[n]*frame[n+p] for n in range(len(frame)-p)) for p in range(order + 1)]
    a_rest = solve_toeplitz((r[:-1],r[:-1]), -np.array(r[1:]))
    a = np.insert(a_rest, 0, 1)
    return a

a_coeffs = lpc_toeplitz(windowed_frame, lpc_order)

w, h_original = freqz(windowed_frame, worN=512, fs=fs)
original_spectrum = 20*np.log10(np.abs(h_original) + 1e-6)

w, h_lpc = freqz(1, a_coeffs, worN=512, fs=fs)
lpc_spectrum = 20*np.log10(np.abs(h_lpc) + 1e-6)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(w, original_spectrum, label='Original Signal Spectrum', color='gray', alpha=0.4)
ax.plot(w, lpc_spectrum, label=f'LPC Spectral Envelope (Order={lpc_order})', color='red')
ax.set_title(f"LPC Spectral Envelope vs Original Spectrum (p={lpc_order})")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.grid(True, linestyle='--')
ax.legend(loc='upper right', fontsize=7)
st.pyplot(fig)

st.info(
"""
Observations:
* p < 6:        Model couldn't catch the details, formant disappear.
* p ~ 10-12:    The ideal range. 3 main formants are perfectly visible.
* p > 40:       Overfitting starts. Model confuses the background noise with the speech.
"""
)