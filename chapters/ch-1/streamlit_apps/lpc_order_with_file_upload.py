
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.io import wavfile

import lpc_brute_force_calculation as lpc_bfc

st.set_page_config(page_title="Understanding LPC Order",
                   page_icon="🦈", layout="wide",
                   initial_sidebar_state="collapsed")
st.title('Understanding LPC Order')
st.info("Please upload a .wav file.")

uploaded_file = st.file_uploader("Upload a speech file (.wav)", type=["wav"])

col1, col2 = st.columns(2)
with col1:
    lpc_order = st.slider("LPC Order (p)", min_value=2, max_value=60, value=12, step=1)

if uploaded_file is not None:

    fs, audio_data = wavfile.read(uploaded_file)

    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0] # mono conversion

    audio_data = audio_data.astype(np.float64)
    if np.max(np.abs(audio_data)) > 0:
        audio_data /= np.max(np.abs(audio_data)) # normalize

    duration = len(audio_data)/fs
    frame_duration = 0.03
    frame_size = int(frame_duration*fs)
    
    with col2:
        start_time = st.slider("Select Frame Start Time (s)", 
                               min_value=0.0, 
                               max_value=max(0.0, duration - frame_duration), 
                               value=min(duration / 2, max(0.0, duration - frame_duration)), 
                               step=0.01)
        
    start_idx = int(start_time*fs)
    frame = audio_data[start_idx : start_idx + frame_size]
    windowed_frame = frame*np.hamming(len(frame))
    a_coeffs, G = lpc_bfc.lpc_calculation(windowed_frame, lpc_order)
    w, h_original = freqz(windowed_frame, worN=512, fs=fs)
    original_spectrum = 20*np.log10(np.abs(h_original) + 1e-6)
    w, h_lpc = freqz(G, a_coeffs, worN=512, fs=fs)
    lpc_spectrum = 20*np.log10(np.abs(h_lpc) + 1e-6)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(w, original_spectrum, label='Original Signal Spectrum', color='gray')
    ax.plot(w, lpc_spectrum, label=f'LPC Spectral Envelope (Order={lpc_order})', color='red')
    ax.set_title(f"LPC Spectral Envelope vs Original Spectrum (p={lpc_order})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    st.pyplot(fig)

st.info(
"""
**Observations for Real Speech:**
* **p < 6:** Model captures only the general spectral slope, failing to isolate individual vocal tract formants.
* **p ~ 10-12:** The ideal range for speech sampled at 8 kHz. The main vocal tract formants are clearly visible.
* **p > 40:** Overfitting occurs. The LPC envelope starts tracing the harmonic fine structure (pitch) and background noise rather than just the vocal tract envelope.
"""
)