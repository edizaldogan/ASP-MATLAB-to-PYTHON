"""
App 2 - Auditory Masking Explorer
==================================
The ONE variable the user controls: the level (in dB SPL) of a 1000 Hz
masker tone.

Concept this teaches
---------------------
This directly reproduces the textbook example from Section 3.1.3 / Fig 3.9:
"A pure tone at 1000 Hz with an intensity of 80 dB SPL will make another
pure tone at 2000 Hz and 40 dB SPL inaudible."

We plot:
  - the absolute auditory threshold (the curve you need to be above to be
    heard at all, in silence)
  - the *masked* threshold produced by the 1000 Hz masker (using the same
    spreading function 'vf' used in MPEG1_psycho_acoustic_model1.m/py)
  - a fixed probe tone at 2000 Hz, 40 dB SPL

As you raise the masker level, watch the masked-threshold curve rise near
1000 Hz and swallow the probe tone.

Run with:
    streamlit run app2_masking_threshold.py
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Auditory Masking Explorer", layout="wide")
st.title("👂 Auditory Masking Explorer")
st.write(
    "This reproduces the book's own example: a masker tone at **1000 Hz** "
    "can hide a quieter probe tone at **2000 Hz / 40 dB SPL**. "
    "Drag the masker level and watch the masking threshold curve rise."
)

# ----------------------------------------------------------------------
# The ONLY variable the user manipulates
# ----------------------------------------------------------------------
masker_level_db = st.slider("Masker tone level at 1000 Hz (dB SPL)", 0, 100, 60)

# ----------------------------------------------------------------------
# Fixed elements
# ----------------------------------------------------------------------
masker_freq = 1000.0
probe_freq = 2000.0
probe_level_db = 40.0

f = np.linspace(20, 20000, 2000)


def bark(freq_hz):
    """Standard Zwicker approximation converting Hz to Bark scale
    (same underlying scale used by Table_z in MPEG1_psycho_acoustic_model1)."""
    return 13 * np.arctan(0.00076 * freq_hz) + 3.5 * np.arctan((freq_hz / 7500) ** 2)


def absolute_threshold_db(freq_hz):
    """Terhardt's approximation of the absolute auditory threshold,
    the same formula used in the book's own plotting code."""
    fk = freq_hz / 1000.0
    return 3.64 * fk ** -0.8 - 6.5 * np.exp(-0.6 * (fk - 3.3) ** 2) + 0.001 * fk ** 4


def vf(dz, X):
    """The exact spreading function used in MPEG1_psycho_acoustic_model1
    (tonal masker case). dz = distance in Bark between masker and the
    frequency being evaluated. X = masker level in dB."""
    dz = np.asarray(dz, dtype=float)
    out = np.empty_like(dz)
    m1 = dz < -1
    m2 = (dz >= -1) & (dz < 0)
    m3 = (dz >= 0) & (dz < 1)
    m4 = dz >= 1
    out[m1] = 17 * (dz[m1] + 1) - (0.4 * X + 6)
    out[m2] = (0.4 * X + 6) * dz[m2]
    out[m3] = -17 * dz[m3]
    out[m4] = -(dz[m4] - 1) * (17 - 0.15 * X) - 17
    return out


z_masker = bark(masker_freq)
z_axis = bark(f)
dz = z_axis - z_masker

# Same tonal-masker offset formula as in MPEG1_psycho_acoustic_model1.py:
# LT_tm = X + (-1.525 - 0.275*z_masker - 4.5) + vf(dz, X)
masking_curve = (
    masker_level_db
    + (-1.525 - 0.275 * z_masker - 4.5)
    + vf(dz, masker_level_db)
)

abs_threshold = absolute_threshold_db(f)
combined_threshold = np.maximum(masking_curve, abs_threshold)

# Is the probe audible?
probe_threshold_at_f = np.interp(probe_freq, f, combined_threshold)
is_masked = probe_level_db < probe_threshold_at_f

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
col1, col2 = st.columns([1, 3])
with col1:
    st.metric("Masker level", f"{masker_level_db} dB SPL @ 1000 Hz")
    st.metric("Probe tone", "40 dB SPL @ 2000 Hz (fixed)")
    if is_masked:
        st.success("🙉 Probe tone is MASKED — inaudible!")
    else:
        st.warning("👂 Probe tone is AUDIBLE.")

with col2:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(f, abs_threshold, "k--", label="Absolute auditory threshold")
    ax.plot(f, combined_threshold, "b", linewidth=2, label="Masked threshold (masker + absolute)")
    ax.axvline(masker_freq, color="tab:orange", linestyle=":", label="Masker (1000 Hz)")
    ax.scatter(
        [probe_freq],
        [probe_level_db],
        color="red" if not is_masked else "green",
        s=100,
        zorder=5,
        label="Probe tone (2000 Hz, 40 dB)",
    )
    ax.set_xscale("log")
    ax.set_xlim(50, 20000)
    ax.set_ylim(-20, 110)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Sound Pressure Level (dB)")
    ax.set_title("Masking threshold around a 1000 Hz masker tone")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

st.info(
    "This is the same spreading function ('vf') and the same offset "
    "formula your `MPEG1_psycho_acoustic_model1` code uses for every "
    "tonal masker it finds — just applied here to a single, isolated tone "
    "so the effect is easy to see."
)
