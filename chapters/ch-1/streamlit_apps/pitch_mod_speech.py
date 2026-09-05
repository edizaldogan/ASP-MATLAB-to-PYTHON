
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
uploaded_file = st.sidebar.file_uploader("Upload a speech.")

# 2.Mic input
with st.sidebar.container(border=True):
    st.markdown("Or record yourself.")
    audio_bytes = audio_recorder(text="Click to record:", icon_size="1x")


# INITIAL TITLE
if uploaded_file is None and audio_bytes is None:
    st.title('Pitch Modification')
    st.markdown("""
    Upload a speech file or record your own speech.\n
    The algorithm will extract the shape of your vocal tract (LPC) for each 30ms frame and replace your vocal cords with a robotic impulse train.\n
    You may manipulate the pitch of your voice by changing the frequency of the impulse train.
    """)

# USER INPUT
elif uploaded_file is not None or audio_bytes is not None:
    target_pitch = st.slider("Synthesized Pitch (F0 in Hz)", min_value=1, max_value=1000, value=120, step=50)

    # UPLOAD
    if uploaded_file is not None:
        original_signal, fs = sf.read(uploaded_file)

    # RECORD
    elif audio_bytes is not None:
        original_signal, fs = sf.read(io.BytesIO(audio_bytes))

    t = np.arange(len(original_signal))/fs
    frame_length = int(0.03*fs)  # 30ms frame length
    synthesized_signal = np.zeros(len(original_signal))

    # Loop trhough the signal in 30ms frames with 10ms hops, calculate LPC coefficients for # # each frame, and synthesize a new signal with the desired pitch.
    for i in range(0, len(original_signal),int(frame_length/3)):
        frame = original_signal[i:i+frame_length]
        if len(frame) < frame_length:
            break
        windowed_frame = frame*np.hamming(len(frame))
        a_coeffs, sigma_squared = lpc_bfc.lpc_calculation(windowed_frame, 10)
        sigma = np.sqrt(sigma_squared)
        
        impulse_interval_new = int(fs/target_pitch)
        source_new = np.zeros(len(frame))
        source_new[::impulse_interval_new] = 1.0
        synthesized_frame = lfilter([sigma], a_coeffs, source_new)
        
        # Replace original with synthesized frame
        synthesized_signal[i:i+len(synthesized_frame)] = synthesized_frame[:len(frame)]

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
    Listen to the difference.\n
    The synthesized signal sounds like "you" but as a robot (monotone).\n
    This proves that the LPC coefficients successfully captured your vocal tract (the filter), but the pure mathematical impulse train lacks the natural breathiness and jitter of real vocal cords.\n
    This limitation is why modern cell phones use CELP instead of pure LPC.
    """
    )
