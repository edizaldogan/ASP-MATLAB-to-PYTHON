"""
App 3 - Sub-band Aliasing / Downsampling Explorer
===================================================
The ONE variable the user controls: the downsampling factor M.

Concept this teaches
---------------------
This reproduces the book's very first sub-band coding demonstration
(Section 3.2.1, Fig 3.12): take a chirp, downsample it by M (keep every
Mth sample), zero-stuff it back up by M, and look at the spectrogram.
Because no anti-aliasing filter is used, "mirror" copies of the chirp
appear -- this is aliasing, the fundamental phenomenon that all the
filter-bank machinery (QMF, PQMF) in the rest of the chapter exists to
cancel out.

Run with:
    streamlit run app3_aliasing_downsampling.py
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram

st.set_page_config(page_title="Aliasing & Downsampling Explorer", layout="wide")
st.title("〰️ Sub-band Aliasing / Downsampling Explorer")
st.write(
    "This is the book's own chirp experiment (Fig 3.12). We downsample a "
    "chirp by a factor **M** (keep every Mth sample) and zero-stuff it "
    "back up, **with no anti-aliasing filter**. Watch how many mirror "
    "copies of the chirp appear as M grows — this is exactly the aliasing "
    "problem that QMF/PQMF filter banks are designed to cancel."
)

# ----------------------------------------------------------------------
# The ONLY variable the user manipulates
# ----------------------------------------------------------------------
M = st.slider("Downsampling factor M", min_value=2, max_value=8, value=2)

# ----------------------------------------------------------------------
# Fixed signal: a 4-second chirp from 0 to 4 kHz at Fs=8kHz, exactly as
# in the book's ASP_mp3.m Section 3.2.1.
# ----------------------------------------------------------------------
Fs = 8000
duration = 4.0
t = np.arange(0, duration, 1 / Fs)
input_signal = chirp(t, f0=0, t1=duration, f1=4000)

# Naive decimation (no anti-alias low-pass filter applied on purpose,
# to make the aliasing visible)
downsampled = input_signal[::M]

# Zero-stuff back up by M (matches the book's upsampled(1:2:...)=2*downsampled
# convention, generalized to arbitrary M; the *M compensates for the
# energy lost by inserting zeros)
upsampled = np.zeros(len(downsampled) * M)
upsampled[::M] = M * downsampled

# ----------------------------------------------------------------------
# Spectrograms
# ----------------------------------------------------------------------
nperseg = 1024
noverlap = 256

f_orig, t_orig, Sxx_orig = spectrogram(
    input_signal, fs=Fs, window="hann", nperseg=nperseg, noverlap=noverlap
)
f_up, t_up, Sxx_up = spectrogram(
    upsampled, fs=Fs, window="hann", nperseg=nperseg, noverlap=noverlap
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original chirp")
    fig1, ax1 = plt.subplots(figsize=(5.5, 5))
    ax1.pcolormesh(t_orig, f_orig, 10 * np.log10(Sxx_orig + 1e-12), shading="auto", cmap="jet")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_ylim(0, Fs / 2)
    st.pyplot(fig1)

with col2:
    st.subheader(f"After ↓{M} then ↑{M} (no filter)")
    fig2, ax2 = plt.subplots(figsize=(5.5, 5))
    ax2.pcolormesh(t_up, f_up, 10 * np.log10(Sxx_up + 1e-12), shading="auto", cmap="jet")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_ylim(0, Fs / 2)
    st.pyplot(fig2)

st.metric("Number of alias images visible", f"{M}")
st.info(
    "With M=2 you get exactly the book's Fig 3.12 (right): one mirror "
    "image, so two sinusoids are audible at any instant. As M grows, "
    "the zero-stuffed spectrum fills up with M-1 extra mirror copies of "
    "the chirp — this is why a real sub-band coder can *never* skip the "
    "analysis/synthesis filters: without them, decimating by 32 would "
    "create 31 audible alias images layered on top of the real signal."
)
