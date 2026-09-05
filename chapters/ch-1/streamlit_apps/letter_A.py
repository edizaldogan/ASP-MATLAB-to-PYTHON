import numpy as np
import soundfile as sf
from scipy.signal import lfilter, freqz
import matplotlib.pyplot as plt

# 1. System Parameters
fs = 8000           # Sampling rate (Hz)
duration = 1.0      # Duration of the sound (seconds)
f0 = 120            # Pitch / Fundamental Frequency (Hz)

# 2. The Recipe for the vowel 'a' (/ɑ/)
# Formant frequencies (Hz) and Bandwidths (Hz)
formants = [730, 1090, 2440]
bandwidths = [50, 110, 170]

# 3. Constructing the Vocal Tract Filter (The Math)
poles = []
for F, B in zip(formants, bandwidths):
    # Calculate radius (damping) and angle (frequency) in the Z-plane
    r = np.exp(-np.pi * B / fs)
    theta = 2 * np.pi * F / fs
    
    # Add complex conjugate pole pairs
    poles.append(r * np.exp(1j * theta))
    poles.append(r * np.exp(-1j * theta))

# np.poly converts roots (poles) into polynomial coefficients (our a_k array!)
a_coeffs = np.real(np.poly(poles))

# 4. Constructing the Vocal Cords (The Source)
t = np.arange(int(fs * duration)) / fs
source = np.zeros(len(t))
impulse_interval = int(fs / f0)
source[::impulse_interval] = 1.0  # The robotic Dirac comb

# 5. Synthesis: Pushing the Source through the Filter
synthesized_a = lfilter([1], a_coeffs, source)

# 6. Normalization (to prevent audio clipping)
synthesized_a = synthesized_a / np.max(np.abs(synthesized_a))

file_path = r"C:\Users\hp\Documents\Erasmus\Project\ASP\Chapter-1\Ch-1\streamlit_apps\synthetic_a.wav"
# Save the result to listen!
sf.write(file_path, synthesized_a, fs)
print("Sound successfully saved as 'synthetic_a.wav'")

# --- Optional: Plot the Frequency Response to verify ---
w, h = freqz(1, a_coeffs, worN=512, fs=fs)
plt.figure(figsize=(8, 4))
plt.plot(w, 20 * np.log10(np.abs(h)))
plt.title("Synthesized 'a' Vocal Tract Envelope")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True, linestyle='--')
plt.show()