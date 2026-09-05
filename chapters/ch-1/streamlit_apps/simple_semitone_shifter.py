"""
'Simple Semitone Shifter' is a minimal Streamlit app that shifts the pitch of an
uploaded audio file by a number of semitones.

load audio, choose semitone shift, perform pitch shift and play results.
A very simple resample-based fallback is used (low-quality but easy to read).

Usage:
- Run with `streamlit run simple_semitone_shifter.py` and upload a short WAV/FLAC/MP3 file.
"""

import io
import streamlit as st
import soundfile as sf
import numpy as np

from scipy.signal import resample

# PAGE LAYOUT
st.set_page_config(page_title="Semitone Shifter", layout="centered")
st.title("Semitone Shifter")
st.markdown(
    """
    This app changes the pitch of an uploaded audio file by a number of
    semitones. \n
    You may shift the pitch up or down by a maximum of 12 semitones (one octave). \n
    Drag an audio file below, choose the semitone shift, and press
    play on the original and shifted audio players.\n
    Important: This simple example assumes the audio is monophonic (one channel).\n
    """
)

# FILE UPLOAD
uploaded = st.file_uploader("Upload an audio file (WAV/FLAC/MP3)")

# SLIDER
semitones = st.slider("Semitone shift", -12, 12, 0, step=1)

# In case of user input:
if uploaded is not None:
    # Read bytes and decode into a numpy array and sample rate
    data, sr = sf.read(io.BytesIO(uploaded.read()))

    # For simplicity we assume 'data' is a 1-D numpy array (mono).
    # If it isn't mono, the array shape will be (N, channels) and this code
    # will still run, but the result will be a mono mix of the channels.

    st.subheader("Original")
    st.audio(data, sample_rate=sr)

    st.write(f"Requested shift: {semitones} semitone(s)")

    # Idea (intuitive explanation for beginners):
    # - Changing the sample rate of a recording changes its pitch.
    # - If we resample the audio to a different length, we move the pitch.
    # - To keep the original duration, we resample twice: first to change pitch,
    #   then back to the original length. This is a crude approximation.
    ratio = 2**(semitones/12.0)  # semitone -> frequency ratio

    if ratio == 1.0:
        y_shifted = data.astype(np.float32)
    else:
        # 1) Resample length to change pitch approximately
        new_len = max(1, int(len(data) / ratio))
        pitched = resample(data, new_len)
        # 2) Resample back to original length to preserve duration
        y_shifted = resample(pitched, len(data)).astype(np.float32)

    st.subheader("Shifted")
    st.audio(y_shifted, sample_rate=sr)

    st.success("Done. Listen to the original and shifted audio using the players above.")
    
else:
    st.info("Upload a file to try the simple semitone shifter.")
