import numpy as np
import soundfile as sf
from scipy.signal import lfilter

# 1. System Parameters
fs = 8000           # Sampling rate
duration = 1.0      # Duration in seconds
f0 = 120            # Pitch / Fundamental Frequency

# 2. Phonetic Recipes for 'o' and 'e'
vowels = {
    'o': {
        'formants': [400, 800, 2600],
        'bandwidths': [50, 80, 130]
    },
    'e': {
        'formants': [530, 1840, 2480],
        'bandwidths': [60, 90, 140]
    }
}

# 3. Constructing the Vocal Cords (The Source - remains identical for both)
t = np.arange(int(fs * duration)) / fs
source = np.zeros(len(t))
impulse_interval = int(fs / f0)
source[::impulse_interval] = 1.0  # Robotic Dirac comb

# 4. Generate and save each vowel
for vowel_name, params in vowels.items():
    poles = []
    # Calculate poles from formants and bandwidths
    for F, B in zip(params['formants'], params['bandwidths']):
        r = np.exp(-np.pi * B / fs)
        theta = 2 * np.pi * F / fs
        poles.append(r * np.exp(1j * theta))
        poles.append(r * np.exp(-1j * theta))
        
    # Get denominator coefficients (a_k)
    a_coeffs = np.real(np.poly(poles))
    
    # Synthesize: Push source through the new filter
    synthesized_audio = lfilter([1], a_coeffs, source)
    
    # Normalize to [-1.0, 1.0]
    synthesized_audio = synthesized_audio / np.max(np.abs(synthesized_audio))
    
    # 5. Save directly to Downloads with absolute path
    file_path = rf"C:\Users\hp\Downloads\synthetic_{vowel_name}.wav"
    sf.write(file_path, synthesized_audio, fs)
    print(f"Ses başarıyla kaydedildi: {file_path}")