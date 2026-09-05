
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, periodogram
import soundfile as sf

import lpc_brute_force_calculation as lpc_bfc

st.set_page_config(page_title="Understanding LPC Order",
                   page_icon="🦈", layout="wide",
                   initial_sidebar_state="expanded")

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload a speech file")

if uploaded_file is None:
    st.title('What happens if we change LPC Order?')
    st.info(
        """
        This program will show if formants are correctly identified for varying 
        LPC orders.
        """)

if uploaded_file is not None:

    lpc_order = st.sidebar.slider("LPC Order (p)", min_value=2, max_value=60, value=12, step=1)

    audio, fs = sf.read(uploaded_file)

    duration = len(audio)/fs
    frame_duration = 0.03
    frame_size = int(frame_duration*fs)
    

    start_time = st.sidebar.slider("Select Frame Start Time (s)", 
                            min_value=0.0, 
                            max_value=max(0.0, duration-frame_duration), 
                            value=min(duration/2, max(0.0, duration - frame_duration)), 
                            step=0.01)
        
    start_idx = int(start_time*fs)
    frame = audio[start_idx:start_idx + frame_size]
    windowed_frame = frame*np.hamming(len(frame))
    a_coeffs, sigma_square = lpc_bfc.lpc_calculation(windowed_frame, lpc_order)
    sigma = np.sqrt(sigma_square)

    w_1, Pxx = periodogram(windowed_frame, fs=2, window='boxcar',detrend=False, nfft=1024)
    original_spectrum_dB = 10*np.log10(np.maximum(Pxx, 1e-10))

    w_2, h_lpc = freqz(1, a_coeffs, worN=513)
    lpc_spectrum = 20*np.log10(sigma*np.abs(h_lpc) + 1e-10)

    # AUDIO PLAYER
    st.caption("Listen to Entire Speech, see below for comments.")
    st.audio(audio, sample_rate=fs)

    # FIGURES
    fig = plt.figure(figsize=(12, 7))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    # Entire speech
    t_full = np.arange(len(audio))/fs
    ax1.plot(t_full, audio, color='blue', linewidth=0.3)
    ax1.axvspan(start_time, start_time+frame_duration, color='red', label='Selected Frame')
    ax1.set_title("Entire Speech Signal", fontsize=11)
    ax1.set_ylabel("Amplitude")
    ax1.margins(x=0)
    ax1.legend(loc='upper right')

    # Zoomed frame
    t_frame = np.arange(frame_size)/fs + start_time
    ax2.plot(t_frame, frame, color='black')
    ax2.set_title(f"Zoom on Selected 30 ms Frame ({start_time:.2f}s - {start_time + frame_duration:.2f}s)")
    ax2.set_ylabel("Amplitude")
    ax2.margins(x=0)

    # Original spectrum and lpc envelope comparison
    ax3.plot(w_1, original_spectrum_dB, label='Original Signal Spectrum')
    ax3.plot(w_2/np.pi, lpc_spectrum, label=f'LPC Spectral Envelope (Order={lpc_order})')
    ax3.set_title(f"LPC Spectral Envelope vs Original Spectrum of the Zoomed Frame (p={lpc_order})")
    ax3.set_xlabel("Frequency (normalized)")
    ax3.set_ylabel("Magnitude (dB)")
    ax3.grid(True, linestyle='--')
    ax3.legend(loc='upper right')

    plt.tight_layout()
    st.pyplot(fig)

    st.info(
    """
    **Observations for Real Speech:**
    * **p < 6:** Model captures only the general spectral slope, failing to isolate individual vocal tract formants.
    * **p ~ 10-12:** The ideal range for speech sampled at 8 kHz. The main vocal tract formants are clearly visible.
    * **p > 40:** Overfitting occurs. The LPC envelope starts tracing the harmonic fine structure (pitch) and background noise rather than just the vocal tract envelope.
    """
    )