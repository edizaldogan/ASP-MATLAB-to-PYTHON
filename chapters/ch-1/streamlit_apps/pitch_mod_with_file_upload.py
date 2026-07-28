import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter, freqz
import soundfile as sf
import io

st.set_page_config(page_title="Understanding Pitch Modification",
                   page_icon="🦈", layout="wide",
                   initial_sidebar_state="collapsed")
st.title('Understanding Pitch Modification (Source-Filter)')

st.markdown("""
Upload a short `.wav` file of a voiced sound (like you saying "Aaaaa"). 
The algorithm will extract the shape of your vocal tract (LPC) and replace your natural vocal cords with a robotic impulse train at a pitch of your choice.
""")

col_ctrl, col_upload = st.columns([1, 1])

with col_ctrl:
    target_pitch = st.slider("Synthesized Pitch (F0 in Hz)", min_value=50, max_value=300, value=120, step=10)
    lpc_order = st.slider("LPC Order (p)", min_value=2, max_value=30, value=12, step=2)

with col_upload:
    uploaded_file = st.file_uploader("Upload a .wav file (Monophonic, Voiced Speech)", type=["wav"])

# Levinson Durbin Algorithm
def lpc_toeplitz(frame, order):
    r = [sum(frame[n] * frame[n+p] for n in range(len(frame)-p)) for p in range(order + 1)]
    a_rest = solve_toeplitz((r[:-1],r[:-1]), -np.array(r[1:]))
    a = np.insert(a_rest, 0, 1)
    return a

if uploaded_file is not None:
    # 1. Read the uploaded audio file
    original_signal, fs = sf.read(uploaded_file)
    
    # Convert to mono if stereo, and cast to float
    if len(original_signal.shape) > 1:
        original_signal = original_signal[:, 0]
    original_signal = original_signal.astype(np.float64)
    
    # Take only the first 2000 samples (~0.25 seconds) to ensure Wide-Sense Stationarity
    if len(original_signal) > 2000:
        original_signal = original_signal[500:5000] 
        
    t = np.arange(len(original_signal)) / fs
    
    # 2. Extract filter (LPC Coefficients) from the REAL voice
    windowed_signal = original_signal * np.hamming(len(original_signal))
    a_coeffs = lpc_toeplitz(windowed_signal, lpc_order)
    
    # 3. Synthesize with ROBOTIC Source at Target Pitch
    impulse_interval_new = int(fs / target_pitch)
    source_new = np.zeros(len(t))
    source_new[::impulse_interval_new] = 1.0  # Robotic Dirac comb
    
    synthesized_signal = lfilter([1], a_coeffs, source_new)
    
    orig_audio_norm = original_signal / np.max(np.abs(original_signal))
    synth_audio_norm = synthesized_signal / np.max(np.abs(synthesized_signal))
    
    # 4. Plotting and Audio Output
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Your Real Voice (Natural Source)")
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.plot(t * 1000, original_signal, color='gray')
        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel("Time (ms)")
        ax1.grid(True, linestyle='--')
        st.pyplot(fig1)
        st.audio(orig_audio_norm, sample_rate=fs)
        
    with col2:
        st.subheader(f"Synthesized Voice (Robotic Source at {target_pitch} Hz)")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(t * 1000, synthesized_signal, color='blue')
        ax2.set_xlabel("Time (ms)")
        ax2.grid(True, linestyle='--')
        st.pyplot(fig2)
        st.audio(synth_audio_norm, sample_rate=fs)

    st.info(
    """
    **Educational Biofeedback:**
    Listen to the difference! The synthesized signal sounds like "you," but as a robot. 
    This proves that the LPC coefficients successfully captured your unique vocal tract (the filter), but the pure mathematical impulse train lacks the natural breathiness and jitter of your real vocal cords. This limitation is why modern cell phones use CELP instead of pure LPC!
    """
    )
else:
    st.warning("👈 Please upload a short .wav file of you saying a vowel to start the experiment!")