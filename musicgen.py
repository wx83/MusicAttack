from transformers import AutoProcessor, MusicgenForConditionalGeneration, Trainer, TrainingArguments
import torch
import math
import torchaudio
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset


torch.cuda.set_per_process_memory_fraction(0.8, device=torch.device('cuda:0'))
torch.backends.cuda.matmul.allow_tf32 = True #
def collate_fn(batch):

    audio_tensors = [torch.tensor(item["audio"]["array"]) for item in batch]
    
    padded_audio_features = pad_sequence(audio_tensors, batch_first=True) # audio has different lengths

    padded_audio_features = padded_audio_features.unsqueeze(1).float()
    
    sampling_rates = [item["audio"]["sampling_rate"] for item in batch]  # List of sampling rates
    text_prompts = [item["text"] for item in batch]  # List of text prompts
    
    return padded_audio_features, sampling_rates, text_prompts


def model_training(train_loader, valid_loader, save_path, num_epochs=10):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model.custom_linear = nn.Linear(2048, 128) # for memory concern

    # Move the new layer to GPU as well
    model.custom_linear = model.custom_linear.to("cuda")
    model = model.to("cuda")
    model.train()


    optimizer = Adam(model.parameters(), lr=5e-5)

    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=2000,
        n_mels=128,  # Set to 2048 to match the feature dimension
        hop_length=128,  # Adjust as necessary to get the correct number of time steps
        n_fft=1024  # Use a higher n_fft to capture more frequency details
    )
    mel_spectrogram_transform = mel_spectrogram_transform.to("cuda")
    mse_loss = torch.nn.MSELoss()

    # feature_reducer = nn.Linear(2048, 128)

    # Training loop
    num_epochs = 10 # Define the number of epochs
    low_val_loss = math.inf
    for epoch in range(num_epochs):
        total_loss = 0

        for audio_features, sampling_rates, text_prompts in train_loader:

            audio_features = audio_features.to("cuda")

            inputs = processor(
                text=text_prompts,
                padding=True,
                return_tensors="pt",
                truncation=True
            ).to("cuda")  # Move text inputs to GPU
            inputs['input_values'] = audio_features # continuation
            outputs = model(**inputs).logits  # Use `last_hidden_state` or appropriate output
            outputs = outputs.float()  # Ensure the output is in float format
            outputs = outputs.to("cuda")

            # take the last 
            # outputs = outputs[-1, :, :].unsqueeze(0) # [1, 250, 2048]
            outputs = outputs.mean(dim=0, keepdim=True)  # average should close to GT
            outputs = model.custom_linear(outputs) # [1, 250, 128]
            outputs = outputs.permute(0, 2, 1) # [1, 128, 250]
            # last column take mean
            generated_mel = outputs.mean(dim=2, keepdim=True)
            print(f"outputs shape = {generated_mel.size()}") # musicgen shape = [1, 128, 250]
            target_mel = mel_spectrogram_transform(audio_features).to("cuda")
            target_mel = target_mel.squeeze(1).mean(dim=2, keepdim=True)
            print(f"target_mel shape = {target_mel.size()}") # target_mel shape = torch.Size([1, 128, 250])

            min_time_steps = min(generated_mel.size(-1), target_mel.size(-1)) // 10
            # print(f"min_time_steps: {min_time_steps}")
            generated_mel = generated_mel[..., :min_time_steps]
            target_mel = target_mel[..., :min_time_steps]

            # Compute the loss
            loss = mse_loss(generated_mel, target_mel)
            loss.backward()
            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()  # Clear gradients for the next step

            # Accumulate loss for monitoring
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")
    # save the trained model
        model.eval()
        for audio_features, sampling_rates, text_prompts in valid_loader:
            audio_features = audio_features.to("cuda")
            inputs = processor(
                text=text_prompts,
                padding=True,
                return_tensors="pt",
                truncation=True
            ).to("cuda")  # Move text inputs to GPU
            inputs['input_values'] = audio_features # continuation
            outputs = model(**inputs).logits  # Use `last_hidden_state` or appropriate output
            outputs = outputs.float()  # Ensure the output is in float format
            outputs = outputs.to("cuda")

            # take the last 
            outputs = outputs[-1, :, :].unsqueeze(0) # [1, 250, 2048]
            outputs = model.custom_linear(outputs) # [1, 250, 128]
            outputs = outputs.permute(0, 2, 1) # [1, 128, 250]

            # last column take mean
            generated_mel = outputs.mean(dim=2, keepdim=True)
            print(f"outputs shape = {generated_mel.size()}") # musicgen shape = [1, 128, 250]
            target_mel = mel_spectrogram_transform(audio_features).to("cuda")
            target_mel = target_mel.squeeze(1).mean(dim=2, keepdim=True)
            print(f"target_mel shape = {target_mel.size()}") # target_mel shape = torch.Size([1, 128, 250])

            min_time_steps = min(generated_mel.size(-1), target_mel.size(-1)) // 5
            # print(f"min_time_steps: {min_time_steps}")
            generated_mel = generated_mel[..., :min_time_steps]
            target_mel = target_mel[..., :min_time_steps]

            # Compute the loss
            loss = mse_loss(generated_mel, target_mel)
            val_loss = loss.item()
            # if val_loss < low_val_loss:
            #     low_val_loss = val_loss
            #     model_save_path = Path(save_path)
            #     model_save_path.mkdir(parents=True, exist_ok=True)  # Create the directory
            #     model.save_pretrained(model_save_path)  # Save the model
        model_save_path = Path(save_path)
        model_save_path.mkdir(parents=True, exist_ok=True)  # Create the directory
        model.save_pretrained(model_save_path)  # Save the model

if __name__ == "__main__":
    dataset_1 = load_dataset("audiofolder", data_dir="/home/users/wx83/GNN_baseline/590/my_dataset") # only load wav in array

    dataset_2 = load_dataset("audiofolder", data_dir="/home/users/wx83/GNN_baseline/590/my_perturb_dataset")

    dataset_train_ori = dataset_1['train']
    dataset_val_ori = dataset_1['validation']
    subset_train_ori = Subset(dataset_train_ori, range(300))  # Use a subset of the training data for faster training
    subset_valid_ori = Subset(dataset_val_ori, range(50))
    train_dataloader_ori = DataLoader(subset_train_ori, batch_size=1, shuffle=True, collate_fn=collate_fn)
    valid_dataloader_ori = DataLoader(subset_valid_ori, batch_size=1, shuffle=False, collate_fn=collate_fn)



    dataset_train_perb= dataset_2['train']
    dataset_val_perb = dataset_2['validation']
    subset_train_perb = Subset(dataset_train_perb, range(300))  # Use a subset of the training data for faster training
    train_dataloader_perb = DataLoader(subset_train_perb, batch_size=1, shuffle=True, collate_fn=collate_fn)
    valid_dataloader_perb = DataLoader(dataset_2['validation'], batch_size=1, shuffle=False, collate_fn=collate_fn)



    # Create two subsets of 150 samples each
    subset1 = Subset(dataset_train_ori, range(150))
    subset2 = Subset(dataset_train_perb, range(150, 300))

    subset1_val= Subset(dataset_val_ori, range(25))
    subset2_val = Subset(dataset_val_perb, range(25))

    # Combine the two subsets into a single dataset
    combined_dataset = ConcatDataset([subset1, subset2])
    combined_dataset_val = ConcatDataset([subset1_val, subset2_val])

    train_dataloader_mix = DataLoader(combined_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    valid_dataloader_mix = DataLoader(combined_dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # model_training(train_dataloader_ori, valid_dataloader_ori, "/home/users/wx83/GNN_baseline/590/model/model_ori")
    # model_training(train_dataloader_perb, valid_dataloader_perb, "/home/users/wx83/GNN_baseline/590/model/model_perb")
    model_training(train_dataloader_mix, valid_dataloader_mix, "/home/users/wx83/GNN_baseline/590/model/model_mix")