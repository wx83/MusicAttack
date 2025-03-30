from transformers import EncodecModel, AutoProcessor, MusicgenForConditionalGeneration, Trainer, TrainingArguments
import torch
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.nn.functional as F
import torchaudio
import pandas as pd
from torch.utils.data import Subset
import os
import torch
import numpy as np
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def load_model():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to("cuda")
    
    # Debug print to check pad_token_id
    print("Initial pad_token_id:", model.config.pad_token_id)
    # source https://huggingface.co/facebook/musicgen-small/blob/main/generation_config.json
    model.config.decoder_start_token_id = 2048

    model.train()  # Set to training mode
    return model, processor



def normalize_audio(waveform):
    """
    Normalize waveform to [-1, 1] range.
    """
    waveform = torch.tensor(waveform, dtype=torch.float32)  # Convert to tensor
    waveform = waveform - waveform.mean()  # Remove DC offset
    waveform = waveform / torch.max(torch.abs(waveform))  # Normalize to [-1, 1]
    return waveform


def preprocess_audio(batch, model, processor):
    """
    Preprocesses audio by normalizing, resampling, and tokenizing it into discrete representations.
    """
    waveform = batch["audio"]["array"]  # e.g., shape: (129545,)
    waveform = normalize_audio(waveform)  # Normalize to [-1, 1]
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=32000)
    waveform = resampler(waveform)  # Resample to 32kHz for the compression model
    compression_model = EncodecModel.from_pretrained("facebook/encodec_32khz")
    compression_model = compression_model.to("cuda")  

    with torch.no_grad():
        inputs = processor(waveform, sampling_rate=32000, return_tensors="pt").to("cuda")
        tokens = compression_model.encode(inputs["input_values"])  # Encode to discrete tokens
    return {"tokens": tokens[0]}  # Select the first token stream

def load_and_process_dataset(model, data_dir, metadata_csv):
    metadata_df = pd.read_csv(metadata_csv)
    metadata_dict = dict(zip(metadata_df["file_name"], metadata_df["text"]))
    
    def add_text_prompt(example):
        file_path = example.get("audio", {}).get("path", None)
        if file_path is None:
            example["text"] = ""
            return example

        # Define the root path that should be removed.
        root_path = Path("/usr/project/xtmp/wx83/my_dataset")
        try:
            # Extract the relative path, e.g. "data/train/_0-2meOf9qY_1.wav"
            relative_path = Path(file_path).relative_to(root_path)
        except ValueError:
            # If the file_path is not under root_path, use it as is.
            relative_path = Path(file_path)

        key = str(relative_path)  # key will be "data/train/_0-2meOf9qY_1.wav"
        # print(f"text search key = {key}")  # Debug: Check the key used for lookup
        example["text"] = metadata_dict.get(key, "")
        print(f"example text = {example['text']}")  # Debug: Check the text prompt added to the example
        return example
    
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    dataset = load_dataset("audiofolder", data_dir=data_dir)['train']
    print(f"load dataset = {dataset[0]}")
    dataset = dataset.map(add_text_prompt, num_proc=1)
    # your custom preprocess_audio function to every single example in your dataset
    dataset = dataset.map(lambda batch: preprocess_audio(batch, model, processor), batched=False)
    dataset = dataset.remove_columns(["label"])

        

    return dataset


def custom_collate_fn(batch):
    # Process tokens: each is assumed to be shape (1, 1, 4, T)
    token_tensors = [torch.tensor(example["tokens"]) for example in batch]
    max_len = max(t.shape[-1] for t in token_tensors)
    
    # Pad each token tensor along the last dimension (time dimension)
    padded_tokens = [F.pad(t, (0, max_len - t.shape[-1])) for t in token_tensors]
    
    # Stack tokens into a single tensor of shape (B, 1, 1, 4, max_len)
    tokens = torch.stack(padded_tokens)
    
    # Collect text prompts from the batch (assuming they are raw strings)
    texts = [example["text"] for example in batch]
    
    return {"tokens": tokens, "text": texts}

def train_model(model, train_loader, processor, num_epochs=10, learning_rate=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    pad_token_id = model.generation_config.pad_token_id  # Use pad token from generation config
    model.train()  # Ensure model is in training mode

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Process text prompt (expects a list of strings)
            text_prompt = batch["text"]
            text_inputs = processor(
                text=text_prompt,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            # Move all text tensors to CUDA
            text_inputs = {k: v.to("cuda") for k, v in text_inputs.items()} # (batch_size, sequence_length)

            # Process audio tokens: original shape [B, 1, 1, 4, T]
            tokens = batch["tokens"].to("cuda")
            # Remove singleton dimensions to get shape [B, T, 4]
            tokens = tokens.squeeze(1).squeeze(1)
            B, num_codebooks, T = tokens.shape

            # Reshape target_music from [B, num_codebooks, T] to [B*num_codebooks, T]
            target_music = tokens.permute(0, 2, 1)  # Permute to [B, T, num_codebooks]

            outputs = model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask", None),
                labels=target_music
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")


if __name__ == "__main__":
    print("Loading model...")
    root_dir = Path("/usr/xtmp/wx83")
    # ORIGIN_DATASET_PATH = Path("/data") # Change this to your dataset path
    # PERTURB_DATASET_PATH = Path("/usr/xtmp/wx83/my_perturb_dataset/data") # Change this to your dataset path
    
    CURRENT_DATASET_PATH = root_dir / "my_dataset" / "data" # Change this to your dataset path
    METADATA_PATH = root_dir / "my_dataset" / "metadata.csv" # Change this to your metadata CSV path
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    OUTPUR_DIR = Path("/home/users/wx83/GNN_baseline/590/model_0228")

    model, processor = load_model()
    print(f" MusicGen model configuration = {model.config.decoder_start_token_id}")
    origin_train_dataset = load_and_process_dataset(model, CURRENT_DATASET_PATH, METADATA_PATH)
    print(f"Loaded origin dataset with {len(origin_train_dataset)} samples.")

    # Define how many samples to use (e.g., 10% of the dataset)
    fraction_to_use = 0.5 # use 50% for training for faster iteration
    num_samples = int(len(origin_train_dataset) * fraction_to_use)
    subset_indices = list(range(num_samples))

    # Create a subset of the dataset
    subset_train_dataset = Subset(origin_train_dataset, subset_indices)
    print(f"Using subset with {len(subset_train_dataset)} samples.")

    # Create DataLoader using the subset
    origin_train_loader = DataLoader(
        subset_train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=custom_collate_fn
    )

    # Train model using the subset
    model = train_model(model, origin_train_loader, processor, num_epochs=NUM_EPOCHS)

    # Save model
    output_path = OUTPUR_DIR / "origin_model.pth"
    save_model(model, output_path)
