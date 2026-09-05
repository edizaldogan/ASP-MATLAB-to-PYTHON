
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import soundfile as sf
import io
from audio_recorder_streamlit import audio_recorder

import lpc_brute_force_calculation as lpc_bfc

st.set_page_config(page_title="Pitch Modification",
                   page_icon="🦈", layout="wide",
                   initial_sidebar_state="collapsed")

# SIDE BAR:
# 1.File Upload
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload a voiced sound.")

# 2.Mic input
with st.sidebar.container(border=True):
    st.markdown("Or record yourself, say AAAAAA...")
    audio_bytes = audio_recorder(text="Click to record:", icon_size="1x")

# INITIAL TITLE
if uploaded_file is None and audio_bytes is None:
    st.title('Pitch Modification')
    st.markdown("""
    Upload the file of a voiced sound.\n
    The algorithm will extract the shape of your vocal tract (LPC) and replace your vocal cords with a robotic impulse train.\n
    Alternatively, press the record button while you say 'AAAA...' and you can then change the pitch of your voice.
    """)

# In case of user input (either file upload or mic input).
elif uploaded_file is not None or audio_bytes is not None:

    target_pitch = st.slider("Synthesized Pitch (F0 in Hz)", min_value=50, max_value=300, value=120, step=10)

    # In case of upload:
    if uploaded_file is not None:
        original_signal, fs = sf.read(uploaded_file)

    # In case of mic input:
    elif audio_bytes is not None:
        original_signal, fs = sf.read(io.BytesIO(audio_bytes))

    # Warning and truncation for long files
    duration = len(original_signal)/fs
    if duration > 2:
        st.warning(f"Audio is {duration:.1f}s long. Truncating to the first 2 seconds for a clear impulse response.")
        original_signal = original_signal[:int(2*fs)]

    t = np.arange(len(original_signal))/fs
    windowed_signal = original_signal*np.hamming(len(original_signal))
    a_coeffs, sigma_squared = lpc_bfc.lpc_calculation(windowed_signal, 10)
    sigma = np.sqrt(sigma_squared)
    
    impulse_interval_new = int(fs/target_pitch)
    source_new = np.zeros(len(t))
    source_new[::impulse_interval_new] = 1.0
    
    synthesized_signal = lfilter([1], a_coeffs, source_new)
    
    orig_audio_norm = original_signal/np.max(np.abs(original_signal))
    synth_audio_norm = synthesized_signal/np.max(np.abs(synthesized_signal))

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Your Real Voice")
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.plot(t*1000, original_signal)
        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel("Time (ms)")
        ax1.grid(True)
        st.pyplot(fig1)
        st.audio(orig_audio_norm, sample_rate=fs)
        
    with col2:
        st.subheader(f"Synthesized Voice ({target_pitch} Hz)")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(t*1000, synthesized_signal)
        ax2.set_xlabel("Time (ms)")
        ax2.grid(True)
        st.pyplot(fig2)
        st.audio(synth_audio_norm, sample_rate=fs)

    st.info(
    """
    Listen to the difference. The synthesized signal sounds like "you" but as a robot (monotone).
    This proves that the LPC coefficients successfully captured your vocal tract (the filter),
    but the pure mathematical impulse train lacks the natural breathiness and jitter of real vocal cords.
    This limitation is why modern cell phones use CELP instead of pure LPC.
    """
    )
