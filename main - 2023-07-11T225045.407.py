import librosa
import numpy as np

# Load the audio file
audio_file = "path/to/your/audio/file.wav"  # Replace with your audio file path
signal, sr = librosa.load(audio_file)

# Extract the spectrogram
hop_length = 512  # Adjust hop length based on your audio
n_fft = 2048  # Adjust FFT size based on your audio
spec = librosa.stft(signal, hop_length=hop_length, n_fft=n_fft)

# Calculate the power spectrogram
spec_power = np.abs(spec) ** 2

# Find the peak frequencies
frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
peak_indices = np.argmax(spec_power, axis=0)
peak_frequencies = frequencies[peak_indices]

# Define a threshold for bird song detection
threshold = 0.1  # Adjust the threshold based on your audio

# Detect bird songs
bird_song_indices = np.where(peak_power > threshold)[0]
bird_song_frequencies = peak_frequencies[bird_song_indices]

# Print the detected bird songs
print("Detected bird songs:")
for frequency in bird_song_frequencies:
    print(frequency)

