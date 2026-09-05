"""
MP3 Compression Explorer:
This app lets the user upload an audio file, pick a target bit rate, and run the
psychoacoustic model + greedy bit allocation algorithm + 32-channel PQMF filter bank on it. The user can then inspect the compression results. User can check whether a difference can be heard when listening the compressed version. User can try to optimize the compression by adjusting the target bit rate.

To run this app, run the following command in the terminal:
streamlit run app.py

Requirements:
MPEG1_bit_allocation.py
MPEG1_psycho_acoustic_model1.py
PQMF32_prototype.py 
matplotlib (for the bits-per-sub-band chart)

The above files must sit in the same folder with this app.
"""

import io
from math import gcd
import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib

from matplotlib.figure import Figure
from scipy.io import wavfile
from scipy.signal import resample_poly

import MPEG1_bit_allocation as mpeg_ba
import MPEG1_psycho_acoustic_model1 as mpeg_pam
from PQMF32_prototype import PQMF32_prototype

st.set_page_config(page_title="MP3 Compression Explorer", layout="wide")

FS = 44100  # required by the reference psychoacoustic model / filter tables
N_TAPS = 512

def build_filterbank():
    hn = np.array(PQMF32_prototype())
    G = np.zeros((32, N_TAPS))
    for i in range(32):
        t2 = ((2*i+1)*np.pi/(2*32))*(np.arange(N_TAPS) + 16)
        G[i, :] = hn*np.cos(t2)
    return G

G_MAT = build_filterbank()

def format_bytes(n_bytes):
    # switches from KB to MB above 1024 KB
    kb = n_bytes / 1024
    if kb < 1024:
        return f"{kb:.1f} KB"
    return f"{kb/1024:.2f} MB"

# SIDEBAR CONTROLS
upload = st.sidebar.file_uploader("Upload an audio file", type=None)
st.sidebar.header("Target bit rate")
target_kbps = st.sidebar.slider("Target bit rate (kbit/s)", 32, 320, 128, step=1)

audio_bytes = upload.getvalue() if upload is not None else None
original_size_bytes = len(audio_bytes) if audio_bytes is not None else 0

if audio_bytes is None:
    st.title("MP3 Compression Explorer")
    st.markdown(
    """
    Upload an audio file, pick a target bit rate. This app shows you the resulting compression and lets you listen the reconstructed audio. Try different bit rates and see where you stop hearing a difference. Try to find the optimal compression by adjusting the bit rate.
    """
    )
    st.info("Upload a file to get started.")
    st.stop()

def load_and_preprocess(audio_bytes, fs_target, n_taps):
    data, native_fs = sf.read(io.BytesIO(audio_bytes), dtype="float64")

    # convert to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    original_duration = len(data)/native_fs

    if native_fs != fs_target:
        g = gcd(int(native_fs), fs_target)
        data_rs = resample_poly(data, fs_target // g, int(native_fs) // g)
    else:
        data_rs = data.copy()

    n_384 = (len(data_rs)//384)*384  # clean multiple of one MPEG-1 frame (384 samples)
    data_rs = data_rs[:n_384]

    if len(data_rs) >= n_taps:
        peak = np.max(np.abs(data_rs)) + 1e-12
        if peak > 1.0:
            data_rs = data_rs/peak  #normalize into the model's expected [-1, 1] range

    return data_rs, native_fs, original_duration

try:
    data_rs, native_fs, original_duration = load_and_preprocess(audio_bytes, FS, N_TAPS)
except Exception:
    st.error(
        "Couldn't decode this file. This app relies on `soundfile`, which "
        "reliably supports WAV / FLAC / OGG. Already-compressed formats like "
        "MP3 may not be readable depending on your system's audio libraries -- "
        "try a WAV file instead."
    )
    st.stop()

if len(data_rs) < N_TAPS:
    st.error("Clip is too short after trimmingfor compression. Please upload a longer file.")
    st.stop()

# ALGORITHM
with st.spinner("Running compression simulation... this can take a while for longer files."):
    n_frames = (len(data_rs) - N_TAPS) // 32 + 1
    windows = np.lib.stride_tricks.sliding_window_view(data_rs, N_TAPS)[::32][:n_frames]
    subbands = windows @ G_MAT.T  # (n_frames, 32)

    n_blocks = n_frames//12
    quantized = np.zeros_like(subbands)
    bits_history = np.zeros((n_blocks, 32))
    total_data_bits = 0
    side_info_bits_per_block = 27*(6 + 4)  # 6-bit scale factor + 4-bit N_bits, 27 used sub-bands

    for bi, k in enumerate(range(0, n_blocks * 12, 12)):
        scale_factors = np.max(np.abs(subbands[k:k + 12, :]), axis=0) + 1e-12

        frame = data_rs[175 + k * 32: 175 + k * 32 + 512]
        if len(frame) < 512:
            frame = np.pad(frame, (0, 512 - len(frame)))
        SMR, *_ = mpeg_pam.MPEG1_psycho_acoustic_model1(frame)

        N_bits, _ = mpeg_ba.MPEG1_bit_allocation(SMR, target_kbps * 1000)
        bits_history[bi, :] = N_bits

        for j in range(32):
            if N_bits[j] != 0:
                alpha = 2 ** (N_bits[j] - 1) / scale_factors[j]
                quantized[k:k + 12, j] = np.floor(alpha * subbands[k:k + 12, j] + 0.5) / alpha
            else:
                quantized[k:k + 12, j] = 0.0

        total_data_bits += 12 * np.sum(N_bits)

    used_frames = n_blocks * 12
    quantized_used = quantized[:used_frames]
    data_used = data_rs[:used_frames * 32]

    frames_out = quantized_used @ G_MAT  # (used_frames, 512)
    out_len = (used_frames - 1)*32 + N_TAPS
    x_tilde = np.zeros(out_len)
    for c in range(16):  # 512 / 32 = 16 non-overlapping phases -> fully vectorizable overlap-add
        seg = frames_out[:, c * 32:(c + 1) * 32]
        start = c * 32
        end = start + used_frames * 32
        x_tilde[start:end] += seg.reshape(-1)

    delay = N_TAPS - 1
    x_tilde_aligned = x_tilde[delay: delay + len(data_used)]
    n_cmp = min(len(data_used), len(x_tilde_aligned))
    orig_cmp, recon_cmp = data_used[:n_cmp], x_tilde_aligned[:n_cmp]

    noise = recon_cmp - orig_cmp
    sig_power = np.mean(orig_cmp**2) + 1e-20
    noise_power = np.mean(noise**2) + 1e-20
    measured_snr = 10*np.log10(sig_power/noise_power)

    total_bits = total_data_bits + n_blocks*side_info_bits_per_block
    duration_s = used_frames*32/FS
    achieved_kbps = total_bits / duration_s / 1000
    compressed_size_bytes = total_bits/8
    compression_ratio = original_size_bytes/compressed_size_bytes
    percent_smaller = 100 * (1 - compressed_size_bytes / original_size_bytes)
    original_bitrate_kbps = original_size_bytes * 8 / 1000 / original_duration

# OUTPUTS
st.header("Compression Results")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Compression ratio", f"{compression_ratio:.2f}x")
c2.metric("Space saved", f"{percent_smaller:.1f}%")
c3.metric("Target bit rate", f"{target_kbps} kbps")
c4.metric("Achieved bit rate", f"{achieved_kbps:.1f} kbps")
c5.metric("Reconstruction SNR", f"{measured_snr:.1f} dB")

st.markdown(
    f"""
| Property | Original file | Compressed (simulated) |
|---|---|---|
| Sample rate | {native_fs} Hz | {FS} Hz |
| Duration | {original_duration:.2f} s | {duration_s:.2f} s |
| Bit rate | {original_bitrate_kbps:.1f} kbps | {achieved_kbps:.1f} kbps |
| Size | {format_bytes(original_size_bytes)} | {format_bytes(compressed_size_bytes)} |
"""
)

st.markdown("#### Average bits allocated per sub-band")
avg_bits = bits_history.mean(axis=0)
band_width_hz = FS / 2 / 32  # each sub-band covers this many Hz, e.g. ~689 Hz at 44.1kHz

fig = Figure(figsize=(10, 3.5))
ax = fig.subplots()
bars = ax.bar(range(32), avg_bits)
for i, v in enumerate(avg_bits):
    ax.text(i, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xlabel("Sub-band index (0 = lowest frequencies, 31 = highest)")
ax.set_ylabel("Bits (avg.)")
ax.set_xticks(range(0, 32, 1))
ax.set_ylim(0, max(avg_bits.max() * 1.3, 1))

# secondary axis on top: converts sub-band index <-> frequency (kHz), so both
# scales stay in sync with the same bars automatically
def band_to_khz(x):
    return x * band_width_hz / 1000

def khz_to_band(f):
    return f * 1000 / band_width_hz

secax = ax.secondary_xaxis('top', functions=(band_to_khz, khz_to_band))
secax.set_xlabel(f"Frequency (kHz) -- each sub-band spans ~{band_width_hz/1000:.2f} kHz")

st.pyplot(fig)
st.markdown(
f"""
- Bars show the number of bits assigned to each sub-band. They are averaged across every 384-sample block of the audio.
- Each sub-band covers a fixed slice of the spectrum, about {band_width_hz:.0f} Hz wide (Fs/2 / 32). Sub-band 0 covers roughly 0-{band_width_hz:.0f} Hz, sub-band 31 covers roughly {31*band_width_hz/1000:.1f}-{32*band_width_hz/1000:.1f} kHz. The top axis shows this mapping directly.
- Within a single block, bit counts are always integer numbers.
- The fractional-looking values in the graph is a result of averaging different integer allocations across many blocks over time. None of the blocks actually gets fractional number of bits.
- Bands averaging 0 bits are fully masked. They are identified as perceptually unimportant throughout the entire file.
""")
    
st.markdown("#### Listen and Compare")

def to_wav_bytes(x, fs):
    x = np.clip(x, -1, 1)
    buf = io.BytesIO()
    wavfile.write(buf, fs, (x*32767).astype(np.int16))
    return buf.getvalue()

colA, colB = st.columns(2)
colA.markdown("**Original**")
colA.audio(to_wav_bytes(orig_cmp, FS), format="audio/wav")
colB.markdown("**Compressed**")
colB.audio(to_wav_bytes(recon_cmp, FS), format="audio/wav")

with st.expander("Click here for comments."):
    st.markdown(
    """
    - **Compression ratio / space saved**: how much smaller the compressed representation is than the file you uploaded, measured in real bytes.
    - **Target vs. achieved bit rate**: the achieved bit rate might differ      slightly from the target bit rate, since bits are handed out in whole units per sub-band and some fixed side-information (scale factors, bit-allocation table) is always spent regardless of content.
    - **Reconstruction SNR**: is a similarity measure between original and reconstructed audio. It is NOT the same as the perceived quality (a lower SNR caused by noise hidden in a masked band can still sound perceptually transparent). So, trust your own ears for quality checking, not just this number.
    - **Bits per sub-band**: it shows where the bit budget is actually used, which will look very different for, a heavy bass line versus a high-pitched melody (flute, violin etc.).
    """)
