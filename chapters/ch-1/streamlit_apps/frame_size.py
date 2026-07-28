import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_toeplitz
from scipy.signal import freqz

st.set_page_config(page_title="Understanding Frame Size",
                   page_icon="🦈", layout="wide",
                   initial_sidebar_state="collapsed")
st.title('Understanding Frame Size (Stationarity)')

frame_size_ms = st.slider("Frame Size (ms)", min_value=5, max_value=150, value=30, step=5)

# Fixed variables
fs = 8000
lpc_order = 12

# Create a non-stationary signal (400 Hz transitioning to 2000 Hz)
total_length = int(fs * 0.2) 
half_length = int(total_length / 2)
t_total = np.arange(total_length) / fs
sig_part1 = np.sin(2 * np.pi * 400 * t_total[:half_length])
sig_part2 = np.sin(2 * np.pi * 2000 * t_total[half_length:])
dynamic_signal = np.concatenate([sig_part1, sig_part2])

# Calculate frame bounds explicitly as integers
frame_samples = int((frame_size_ms / 1000.0) * fs)
start_idx = int((total_length / 2) - (frame_samples / 2))
end_idx = start_idx + frame_samples

extracted_frame = dynamic_signal[start_idx:end_idx]
windowed_frame = extracted_frame * np.hamming(len(extracted_frame))

# Levinson Durbin Algorithm
def lpc_toeplitz(frame, order):
    r = [sum(frame[n] * frame[n+p] for n in range(len(frame)-p)) for p in range(order + 1)]
    a_rest = solve_toeplitz((r[:-1],r[:-1]), -np.array(r[1:]))
    a = np.insert(a_rest, 0, 1)
    return a

a_coeffs_frame = lpc_toeplitz(windowed_frame, lpc_order)

w, h_frame = freqz(windowed_frame, worN=512, fs=fs)
frame_spectrum = 20 * np.log10(np.abs(h_frame) + 1e-6)
w, h_lpc = freqz(1, a_coeffs_frame, worN=512, fs=fs)
lpc_spectrum = 20 * np.log10(np.abs(h_lpc) + 1e-6)

# Align energy for visual comparison
lpc_spectrum += (np.max(frame_spectrum) - np.max(lpc_spectrum))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(w, frame_spectrum, label='Actual Frame Spectrum', color='gray', alpha=0.4)
ax.plot(w, lpc_spectrum, label=f'LPC Envelope', color='red')
ax.set_title(f"Spectral Envelope Analysis (Window = {frame_size_ms} ms)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.grid(True, linestyle='--')
ax.legend(loc='upper right', fontsize=7)
st.pyplot(fig)

st.info(
"""
Observations:
* 20-30 ms: The envelope perfectly tracks the formants. The signal is Wide-Sense Stationary (WSS) within this short window.
* > 80 ms: The window is too wide and captures the transition between two different frequencies. The LPC model attempts to average them, ruining the envelope.
"""
)