from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Load MP3 file (make sure FFmpeg is installed and in your PATH)
audio = AudioSegment.from_mp3("BTS (방탄소년단) 'Euphoria _ Theme of LOVE YOURSELF 起 Wonder'.mp3")

# Convert stereo to mono and get raw samples
audio = audio.set_channels(1)
samples = np.array(audio.get_array_of_samples()).astype(np.float32)

# Define convolution kernel
kernel = np.array([1, 0, 1, 0, 1], dtype=np.float32)

# Perform convolution
convoluted = convolve(samples, kernel, mode='same')

# Normalize to [-1, 1] range

convoluted = convoluted / np.max(np.abs(convoluted))

# Scale to int16 for WAV export
convoluted_int16 = (convoluted * 32767).astype(np.int16)

# Save convoluted audio using PyDub
convoluted_audio = AudioSegment(
    convoluted_int16.tobytes(),
    frame_rate=audio.frame_rate,
    sample_width=2,  # 2 bytes = 16 bits
    channels=1
)
convoluted_audio.export("output_convoluted.wav", format="wav")
print("Convoluted audio saved as 'output_convoluted.wav'.")

# Plot original and convoluted audio
samples_norm = samples / np.max(np.abs(samples))  # normalize original for plotting

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(samples_norm, color='blue')
plt.title("Original Audio Signal")

plt.subplot(2, 1, 2)
plt.plot(convoluted, color='red')
plt.title("Convoluted Audio Signal with Kernel [1, 0, 1, 0, 1]")

plt.tight_layout()
plt.show()
