import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
class SpeechCommandsDataset(Dataset):
    def __init__(self, dataset_dir, commands, split="train", sample_rate=16000, n_fft=318, hop_length=149, target_shape=(160, 101)):
        self.dataset_dir = dataset_dir
        self.commands = commands
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_shape = target_shape
        
        # Load split files
        validation_files = set()
        testing_files = set()
        
        validation_list_path = os.path.join(dataset_dir, "validation_list.txt")
        testing_list_path = os.path.join(dataset_dir, "testing_list.txt")
        
        
        with open(validation_list_path, "r") as f:
            validation_files = {os.path.normpath(os.path.join(dataset_dir, line.strip())) for line in f if line.strip()}
        with open(testing_list_path, "r") as f:
            testing_files = {os.path.normpath(os.path.join(dataset_dir, line.strip())) for line in f if line.strip()}
        
        # Collect files for the specified split
        self.files = []
        self.labels = []
        for command_idx, command in enumerate(commands):
            command_dir = os.path.join(dataset_dir, command)
            if not os.path.exists(command_dir):
                print(f"Warning: Directory {command_dir} not found, skipping.")
                continue
            for file_name in os.listdir(command_dir):
                if file_name.endswith(".wav"):
                    full_path = os.path.normpath(os.path.join(command_dir, file_name))
                    if split == "train" and full_path not in validation_files and full_path not in testing_files:
                        self.files.append(full_path)
                        self.labels.append(command_idx)
                    elif split == "validation" and full_path in validation_files:
                        self.files.append(full_path)
                        self.labels.append(command_idx)
                    elif split == "test" and full_path in testing_files:
                        self.files.append(full_path)
                        self.labels.append(command_idx)
        
        print(f"Found {len(self.files)} audio files for {split} split.")

    def __getitem__(self, idx):
        file_path = self.files[idx]
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        if waveform.shape[1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
        elif waveform.shape[1] > 16000:
            waveform = waveform[:, :16000]
        
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2
        )(waveform)
        
        spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        spectrogram_norm = (spectrogram_db - spectrogram_db.mean()) / (spectrogram_db.std() +1e-9)
        
        padded_spectrogram = torch.zeros(1,*self.target_shape)
        rows = min(spectrogram_norm.shape[1], self.target_shape[0])
        cols = min(spectrogram_norm.shape[2], self.target_shape[1])
        padded_spectrogram[:, :rows, :cols] = spectrogram_norm[:, :rows, :cols]
        
        return padded_spectrogram, self.labels[idx]

    def __len__(self):
        return len(self.files)

# Define commands (30 classes)

def load_command_data(batch_size=100):
    commands = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "house", "marvin", "sheila", "tree", "wow", "happy"
    ]

    # Initialize datasets
    dataset_dir = r"C:\Users\Dustin\Documents\Subjects\Application of Data Science\Mixup\data\speech_commands_dataset"
    try:
        train_dataset = SpeechCommandsDataset(
            dataset_dir=dataset_dir,
            commands=commands,
            split="train"
        )
        validation_dataset = SpeechCommandsDataset(
            dataset_dir=dataset_dir,
            commands=commands,
            split="validation"
        )
        test_dataset = SpeechCommandsDataset(
            dataset_dir=dataset_dir,
            commands=commands,
            split="test"
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader

# # --- Plot the spectrogram ---
# train_loader, test_loader, val_loader = load_command_data(batch_size=1)
# commands = [
#     "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
#     "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
#     "bed", "bird", "cat", "dog", "house", "marvin", "sheila", "tree", "wow", "happy"
# ]

# for spec, label in train_loader.dataset:
#     if commands[label] == "yes":
#         yes_spec = spec.squeeze(0).numpy()  # shape (160, 101)
#         break


# plt.figure(figsize=(8, 4))
# plt.imshow(yes_spec, aspect='auto', origin='lower', cmap='magma')
# plt.colorbar(label="Normalized dB")
# plt.title("Spectrogram Example: 'yes'")
# plt.xlabel("Time Frames")
# plt.ylabel("Frequency Bins")
# plt.tight_layout()
# plt.show()
# plt.show()