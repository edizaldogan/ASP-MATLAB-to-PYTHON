import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter

st.set_page_config(page_title="Understanding Pitch Modification",
                   page_icon="🦈", layout="wide",
                   initial_sidebar_state="collapsed")
st.title('Understanding Pitch Modification (Source-Filter)')

target_pitch = st.slider("Target Pitch (F0 in Hz)", min_value=50, max_value=300, value=120, step=1 )

fs = 8000
duration = 0.5  # Increased duration slightly to 0.5 seconds so we can hear it better
t = np.arange(int(duration * fs)) / fs

# Fixed variables
lpc_order = 12
orig_pitch = 100

# 1. Generate original signal (Source + Filter)
impulse_interval_orig = int(fs / orig_pitch)
source_orig = np.zeros(len(t))
source_orig[::impulse_interval_orig] = 1.0

# Vowel-like formants at 500 Hz and 1500 Hz
vowel_wave = np.sin(2 * np.pi * 500 * t) * np.exp(-50*t) + 0.5 * np.sin(2 * np.pi * 1500 * t) * np.exp(-50*t)
original_signal = lfilter([1], [1, -0.9], np.convolve(source_orig, vowel_wave, mode='same'))
windowed_signal = original_signal * np.hamming(len(original_signal))

# Levinson Durbin Algorithm
def lpc_toeplitz(frame, order):
    r = [sum(frame[n] * frame[n+p] for n in range(len(frame)-p)) for p in range(order + 1)]
    a_rest = solve_toeplitz((r[:-1],r[:-1]), -np.array(r[1:]))
    a = np.insert(a_rest, 0, 1)
    return a

# 2. Extract filter (LPC Coefficients)
a_coeffs = lpc_toeplitz(windowed_signal, lpc_order)

# 3. Synthesize with new pitch (New Source + Old Filter)
impulse_interval_new = int(fs / target_pitch)
source_new = np.zeros(len(t))
source_new[::impulse_interval_new] = 1.0
synthesized_signal = lfilter([1], a_coeffs, source_new)

# --- Normalization for Audio Playback ---
# Scale arrays to [-1.0, 1.0] to prevent clipping when Streamlit converts them to audio
orig_audio_norm = original_signal / np.max(np.abs(original_signal))
synth_audio_norm = synthesized_signal / np.max(np.abs(synthesized_signal))

# 4. Plotting and Audio Output
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Original Signal ({orig_pitch} Hz)")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    # Plotting only the first 100ms so the graph isn't too cluttered
    ax1.plot(t[:800] * 1000, original_signal[:800], color='gray')
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (ms)")
    ax1.grid(True, linestyle='--')
    st.pyplot(fig1)
    
    # Built-in Streamlit audio player
    st.audio(orig_audio_norm, sample_rate=fs)

with col2:
    st.subheader(f"Synthesized Signal ({target_pitch} Hz)")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.plot(t[:800] * 1000, synthesized_signal[:800], color='blue')
    ax2.set_xlabel("Time (ms)")
    ax2.grid(True, linestyle='--')
    st.pyplot(fig2)
    
    # Built-in Streamlit audio player
    st.audio(synth_audio_norm, sample_rate=fs)

st.info(
"""
Observations:
* Hit **Play** on both audio players. You will hear the fundamental frequency (pitch) change, but the "vowel" sound remains the same.
* The distance between the peaks (impulses) changes as you move the slider, representing a change in the vocal cords' vibration rate.
* The internal shape of each peak (the resonance) remains identical because the mathematical filter (LPC coefficients) was kept constant.
"""
)