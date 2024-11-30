# audio attack
import torchaudio
import torch
import pathlib
from pathlib import Path

# get a list of file names
root_dir = Path("")
file_name_list = Path("")
# Load the audio file
waveform, sample_rate = torchaudio.load("path_to_your_audio_file.wav")

# 1. Apply Gaussian Noise
def add_gaussian_noise(waveform, mean=0.0, std=0.005):
    noise = torch.randn(waveform.size()) * std + mean
    return waveform + noise

waveform = add_gaussian_noise(waveform)

# 2. Apply Time Stretch
time_stretch = torchaudio.transforms.TimeStretch()
# Note: The hop_length should be set to your preference
waveform = time_stretch(waveform, rate=1.2)  # Adjust 'rate' for speed

# 3. Apply Pitch Shift
pitch_shift = torchaudio.transforms.PitchShift(sample_rate=sample_rate, n_steps=2)  # Adjust n_steps as needed
waveform = pitch_shift(waveform)

# Save the modified audio (optional)
torchaudio.save("transformed_audio.wav", waveform, sample_rate)





# output metadata at:/home/users/wx83/GNN_baseline/590/my_dataset/metadata.csv
# file_name: path, text: description


