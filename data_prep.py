import json
import pandas as pd
import pathlib 
from pathlib import Path
import torch
file_path = '/home/users/wx83/GNN_baseline/590/metadata.json'

data_dicts = []

# # Read and parse each line as a separate JSON object
with open(file_path) as f:
    for line in f:
        data_dicts.append(json.loads(line))

for item in data_dicts:
    # Updating the location path from 'data_aug2/' to 'data/train/'
    item["location"] = item["location"].replace("data_aug2/", "data/validation/")



df_extracted = pd.DataFrame({
    "file_name": [item["location"] for item in data_dicts],
    "text": [item["main_caption"] for item in data_dicts]
})

df_extracted.to_csv('/home/users/wx83/GNN_baseline/590/my_dataset/metadata_validation.csv', index=False)


with open("/home/users/wx83/GNN_baseline/590/train.txt", "r") as f:
    train_files = f.readlines()
train_files = [file.strip() for file in train_files]
df = pd.read_csv('/home/users/wx83/GNN_baseline/590/my_dataset/metadata.csv')

df_save = pd.DataFrame(columns=['file_name', 'text'])

for file in train_files:
    # Check if the file name exists in the 'file_name' column of df
    # if file not in df['file_name'].values:
    if df[df['file_name'] == file] is None:
        print(file)
        row = pd.DataFrame({'file_name': [file], 'text': ['.']})
    else:
        row = df.loc[df['file_name'] == file]
    # Concatenate the new row with df_save
    df_save = pd.concat([df_save, row], ignore_index=True)

# df_train = df[df['file_name'].isin(train_files)]

df_save.to_csv('/home/users/wx83/GNN_baseline/590/my_dataset/metadata_train.csv', index=False)



with open("/home/users/wx83/GNN_baseline/590/validation.txt", "r") as f:
    train_files = f.readlines()
train_files = [file.strip() for file in train_files]
df = pd.read_csv('/home/users/wx83/GNN_baseline/590/metadata_validation.csv')

df_save = pd.DataFrame(columns=['file_name', 'text'])

for file in train_files:
    # Check if the file name exists in the 'file_name' column of df
    if file not in df['file_name'].values:
        row = pd.DataFrame({'file_name': [file], 'text': ['.']})
    else:
        row = df.loc[df['file_name'] == file]

    # Concatenate the new row with df_save
    df_save = pd.concat([df_save, row], ignore_index=True)

# df_train = df[df['file_name'].isin(train_files)]

df_save.to_csv('/home/users/wx83/GNN_baseline/590/my_dataset/metadata_validation.csv', index=False)


with open("/home/users/wx83/GNN_baseline/590/my_dataset/validation.txt", "r") as f:
    test_files = f.readlines()

test_files = [file.strip() for file in test_files]

df_2 = pd.read_csv('/home/users/wx83/GNN_baseline/590/my_dataset/metadata_validation.csv')

df_test = df_2[df_2['file_name'].isin(test_files)]
df_test.to_csv('/home/users/wx83/GNN_baseline/590/my_dataset/metadata_test.csv', index=False)

import torchaudio
import torchaudio.transforms as T
import torch
from pathlib import Path

input_dir = Path("/home/users/wx83/GNN_baseline/590/my_dataset/data/validation")
output_dir = Path("/home/users/wx83/GNN_baseline/590/perturb/validation")

# Load all train files
with open("/home/users/wx83/GNN_baseline/590/valid_file.txt", "r") as f:
    valid_file = f.readlines()
valid_file = [file.strip() for file in valid_file]

# Define parameters for transformations
noise_factor = 0.005  # Intensity of Gaussian noise
time_stretch_rate = 1.1  # Factor for time stretching
pitch_shift_steps = 2  # Number of semitones for pitch shifting

# Process each file one at a time
for file in valid_file:
    try:
        file_path = input_dir / file
        output_path = output_dir / file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)

        # Apply Gaussian noise
        noise = torch.randn_like(waveform) * noise_factor
        waveform_noisy = waveform + noise

        # Downsample audio to reduce memory usage (if sample rate is high)
        if sample_rate > 8000:
            resample_transform = T.Resample(orig_freq=sample_rate, new_freq=8000)
            waveform_noisy = resample_transform(waveform_noisy)
            sample_rate = 8000

        # Apply pitch shift
        pitch_shift = T.PitchShift(sample_rate=sample_rate, n_steps=pitch_shift_steps)
        waveform_shifted = pitch_shift(waveform_noisy)

        # Save the transformed audio
        torchaudio.save(output_path, waveform_shifted, sample_rate)

        # Free up memory
        del waveform, waveform_noisy, waveform_shifted
        torch.cuda.empty_cache()  # If using a GPU
    except Exception as e:
        print(f"Error processing {file}: {e}")
file_path = '/home/users/wx83/GNN_baseline/590/my_dataset/metadata.csv'
metadata = pd.read_csv(file_path)

metadata["file_name"] = metadata["file_name"].str.replace("data", "perturb")

# save
metadata.to_csv('/home/users/wx83/GNN_baseline/590/metadata_perturb.csv', index=False)

