import io

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from scipy.signal import resample_poly
from scipy.signal import resample

st.set_page_config(
    page_title="Instrument Semitone Shifter",
    page_icon="🎻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def to_mono(signal):
    if signal.ndim == 1:
        return signal.astype(np.float64)
    if signal.ndim == 2:
        return np.mean(signal, axis=1).astype(np.float64)
    return signal.reshape(-1).astype(np.float64)


def naive_pitch_shift(signal, sr, semitones):

    if len(signal) == 0:
        return signal

    ratio = 2**(semitones / 12.0)
    if abs(ratio - 1.0) < 1e-12:
        out = np.array(signal, dtype=np.float64)
        if np.max(np.abs(out)) > 0:
            out = out / np.max(np.abs(out))
        return out

    def _stft(y, n_fft=2048, hop_length=None):
        if hop_length is None:
            hop_length = n_fft // 4
        y = np.concatenate([np.zeros(n_fft // 2), y, np.zeros(n_fft // 2)])
        n_frames = 1 + (len(y) - n_fft) // hop_length
        frames = np.stack([y[i * hop_length : i * hop_length + n_fft] for i in range(n_frames)], axis=1)
        win = np.hanning(n_fft)[:, None]
        stft_matrix = np.fft.rfft(frames * win, axis=0)
        return stft_matrix, n_fft, hop_length, win

    def _istft(stft_matrix, n_fft, hop_length, win, expected_len=None):
        n_bins, n_frames = stft_matrix.shape
        y_len = n_frames * hop_length + n_fft
        y = np.zeros(y_len)
        win_sq = win[:, 0] ** 2
        norm = np.zeros(y_len)
        for i in range(n_frames):
            frame = np.fft.irfft(stft_matrix[:, i], n=n_fft).real
            start = i * hop_length
            y[start : start + n_fft] += frame * win[:, 0]
            norm[start : start + n_fft] += win_sq
        nz = norm > 1e-8
        y[nz] /= norm[nz]
        # Remove padding
        y = y[n_fft // 2 : -n_fft // 2]
        if expected_len is not None:
            if len(y) < expected_len:
                y = np.concatenate([y, np.zeros(expected_len - len(y))])
            else:
                y = y[:expected_len]
        return y

    def _phase_vocoder(D, rate, n_fft, hop_length):
        # D: (n_bins, n_frames)
        time_steps = np.arange(0, D.shape[1], rate)
        n_bins = D.shape[0]
        omega = 2.0 * np.pi * np.arange(n_bins) / n_fft
        phi = np.angle(D[:, 0])
        out = np.zeros((n_bins, len(time_steps)), dtype=np.complex128)
        for t_idx, t in enumerate(time_steps):
            t_i = int(np.floor(t))
            frac = t - t_i
            if t_i + 1 < D.shape[1]:
                S1 = D[:, t_i]
                S2 = D[:, t_i + 1]
            else:
                S1 = D[:, t_i]
                S2 = S1
            mag = (1.0 - frac) * np.abs(S1) + frac * np.abs(S2)
            # phase increment estimate
            delta = np.angle(S2) - np.angle(S1)
            # Remove expected phase advance
            delta = delta - omega * hop_length
            # Wrap to [-pi, pi]
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
            # Advance phase
            phi = phi + omega * hop_length + delta
            out[:, t_idx] = mag * np.exp(1j * phi)
        return out

    stretch_factor = 1.0 / ratio
    vocoder_rate = ratio

    n_fft = 2048
    hop_length = n_fft // 4
    D, n_fft_ret, hop_length_ret, win = _stft(signal, n_fft=n_fft, hop_length=hop_length)

    D_stretched = _phase_vocoder(D, rate=vocoder_rate, n_fft=n_fft, hop_length=hop_length)

    expected_len = max(1, int(len(signal) * stretch_factor))
    y_stretched = _istft(D_stretched, n_fft=n_fft, hop_length=hop_length, win=win, expected_len=expected_len)

    y_final = resample(y_stretched, len(signal))

    if np.max(np.abs(y_final)) > 0:
        y_final = y_final / np.max(np.abs(y_final))

    return y_final

def pitch_shift_instrument(signal, sr, semitones):
    mono = to_mono(signal)
    return naive_pitch_shift(mono, sr, semitones)

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload an instrument recording")

with st.sidebar.container(border=True):
    st.caption("Or record directly")
    audio_bytes = audio_recorder(text="Click to record", icon_size="1x")

if uploaded_file is None and audio_bytes is None:
    st.title("Song Semitone Shifter")
    st.markdown(
        """
        Upload a guitar, violin, piano, or other instrument recording.\n
        The app shifts the pitch by a chosen number of semitones while preserving the musical character of the performance.
        A value of +12 semitones means one octave up; -12 means one octave down.
        """
    )
else:
    if uploaded_file is not None:
        raw_audio, fs = sf.read(uploaded_file)
    else:
        raw_audio, fs = sf.read(io.BytesIO(audio_bytes))

    if raw_audio is None or len(raw_audio) == 0:
        st.error("The audio file is empty or unreadable.")
        st.stop()

    original = raw_audio.astype(np.float64)
    mono_original = to_mono(original)
    if np.max(np.abs(mono_original)) > 0:
        mono_original = mono_original / np.max(np.abs(mono_original))

    semitone_shift = st.slider(
        "Pitch shift (semitones)",
        min_value=-12,
        max_value=12,
        value=0,
        step=1,
    )

    shifted = pitch_shift_instrument(mono_original, fs, semitone_shift)
    if np.max(np.abs(shifted)) > 0:
        shifted = shifted / np.max(np.abs(shifted))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.audio(mono_original, sample_rate=fs)
        fig1, ax1 = plt.subplots(figsize=(8, 2.8))
        ax1.plot(np.arange(len(mono_original)) / fs, mono_original)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        st.pyplot(fig1)

    with col2:
        st.subheader(f"Shifted ({semitone_shift:+d} semitones)")
        st.audio(shifted, sample_rate=fs)
        fig2, ax2 = plt.subplots(figsize=(8, 2.8))
        ax2.plot(np.arange(len(shifted)) / fs, shifted)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True)
        st.pyplot(fig2)

    st.subheader("Spectrogram comparison")
    fig3, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    for ax, signal, title in [
        (axes[0], mono_original, "Original"),
        (axes[1], shifted, f"Shifted ({semitone_shift:+d} st)")
    ]:
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1 / fs)
        ax.specgram(signal, Fs=fs, cmap="magma")
        ax.set_title(title)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")

    st.pyplot(fig3)

    st.info(
        """
        For best results on real musical recordings, librosa is preferred. The fallback mode is a quick approximation for instruments, and is useful when the audio is mostly harmonic and stable.
        """
    )
