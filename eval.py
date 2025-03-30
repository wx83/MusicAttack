# generate sample
from transformers import AutoProcessor, MusicgenForConditionalGeneration, Trainer, TrainingArguments
import torch
import math
import torchaudio
from pathlib import Path
import scipy.io.wavfile

def generation(model_path, text_prompt, save_dir, save_name):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained(model_path)
    model = model.to("cuda")
    inputs = processor(
        text=[text_prompt],
        padding=True,
        return_tensors="pt",
    )
    # Move the inputs to GPU
    inputs = {key: value.to("cuda") for key, value in inputs.items()}


    audio_values = model.generate(**inputs, max_new_tokens=256)


    sampling_rate =32000  # Example sampling rate, adjust as needed

    # Move the tensor to the CPU and convert to a NumPy array
    audio_values_cpu = audio_values[0, 0].cpu().numpy()

    # Write the audio data to a WAV file
    save_path = Path(save_dir) / save_name
    scipy.io.wavfile.write(save_path, rate=sampling_rate, data=audio_values_cpu)


# objective evaluation

if __name__ == "__main__":

    with open("/home/users/wx83/GNN_baseline/590/prompts.txt", "r") as f:
        text_prompt = f.readlines()
    text_prompt = [text.strip() for text in text_prompt]

    model_path = Path("/home/users/wx83/GNN_baseline/590/model/model_ori")
    # text_prompt = ["This song contains digital drums playing a simple groove along with two guitars. One strumming chords along with the snare the other one playing a melody on top. An e-bass is playing the footnote while a piano is playing a major and minor chord progression. A trumpet is playing a loud melody alongside the guitar. All the instruments sound flat and are being played by a keyboard. There are little bongo hits in the background panned to the left side of the speakers. Apart from the music you can hear eating sounds and a stomach rumbling. This song may be playing for an advertisement. The chord progression in this song is Bb6. The beat is 3. The song is played at the pace of Presto. The key is Bb major."]
    for idx, text in enumerate(text_prompt):
        save_dir = model_path / "output"
        save_dir.mkdir(parents=True, exist_ok=True)
        count = idx + 1
        save_name = f"{count}.wav"
        generation(model_path, text, save_dir, save_name)

    model_path = Path("/home/users/wx83/GNN_baseline/590/model/model_mix")
    # text_prompt = ["This song contains digital drums playing a simple groove along with two guitars. One strumming chords along with the snare the other one playing a melody on top. An e-bass is playing the footnote while a piano is playing a major and minor chord progression. A trumpet is playing a loud melody alongside the guitar. All the instruments sound flat and are being played by a keyboard. There are little bongo hits in the background panned to the left side of the speakers. Apart from the music you can hear eating sounds and a stomach rumbling. This song may be playing for an advertisement. The chord progression in this song is Bb6. The beat is 3. The song is played at the pace of Presto. The key is Bb major."]
    for idx, text in enumerate(text_prompt):
        save_dir = model_path / "output"
        save_dir.mkdir(parents=True, exist_ok=True)
        count = idx + 1
        save_name = f"{count}.wav"
        generation(model_path, text, save_dir, save_name)

    model_path = Path("/home/users/wx83/GNN_baseline/590/model/model_perb")
    # text_prompt = ["This song contains digital drums playing a simple groove along with two guitars. One strumming chords along with the snare the other one playing a melody on top. An e-bass is playing the footnote while a piano is playing a major and minor chord progression. A trumpet is playing a loud melody alongside the guitar. All the instruments sound flat and are being played by a keyboard. There are little bongo hits in the background panned to the left side of the speakers. Apart from the music you can hear eating sounds and a stomach rumbling. This song may be playing for an advertisement. The chord progression in this song is Bb6. The beat is 3. The song is played at the pace of Presto. The key is Bb major."]
    for idx, text in enumerate(text_prompt):
        save_dir = model_path / "output"
        save_dir.mkdir(parents=True, exist_ok=True)
        count = idx + 1
        save_name = f"{count}.wav"
        generation(model_path, text, save_dir, save_name)
