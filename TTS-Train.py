import os
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import Resample

class TTSDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_sample_rate=22050):
        self.annotations = pd.read_csv(csv_file, delimiter='|', quotechar='"', on_bad_lines='skip', header=None)
        self.root_dir = root_dir
        self.target_sample_rate = target_sample_rate
        self.valid_indices = self._filter_valid_indices()

    def _filter_valid_indices(self):
        valid_indices = []
        for index in range(len(self.annotations)):
            audio_filename = self.annotations.iloc[index, 0].strip() + '.flac'
            audio_path = os.path.join(self.root_dir, audio_filename)
            if os.path.isfile(audio_path):
                valid_indices.append(index)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        actual_index = self.valid_indices[index]
        row = self.annotations.iloc[actual_index]
        audio_filename = row[0].strip() + '.flac'
        audio_path = os.path.join(self.root_dir, audio_filename)
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)(waveform)
            text = row[1].strip()
            return waveform, text
        except Exception as e:
            print(f"Error loading {audio_filename}: {e}")
            raise

def pad_collate(batch):
    (waveforms, texts) = zip(*batch)
    waveforms = [waveform.squeeze(0) for waveform in waveforms] 
    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0)
    return waveforms_padded, texts

class SimpleTacotron2(nn.Module):
    def __init__(self):
        super(SimpleTacotron2, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, waveform):
        x, _ = self.lstm(waveform)
        x = self.fc(x)
        return x

def train(model, data_loader, epochs, optimizer, criterion):
    model.train()
    for epoch in range(epochs):
        for waveforms, texts in data_loader:
            optimizer.zero_grad()
            waveforms = waveforms.unsqueeze(-1) 
            try:
                output = model(waveforms)
                loss = criterion(output, waveforms)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            except RuntimeError as e:
                print(f"Runtime error during training: {e}")

csv_file = '/Users/baristuzemen/tts_data/metadata.csv'
audio_directory = '/Users/baristuzemen/tts_data'

dataset = TTSDataset(csv_file=csv_file, root_dir=audio_directory)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=pad_collate)

model = SimpleTacotron2()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

try:
    train(model, data_loader, 2, optimizer, criterion)
except RuntimeError as e:
    print(f"Runtime error: {e}")
