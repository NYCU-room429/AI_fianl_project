import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import utils_ai
import os

class CachedMelLabelDataset(Dataset):
    def __init__(self, track_list, mel_dir, instrument_classes, instruments_mapping, n_mels=128, max_frames=1000, train=False):
        self.track_list = track_list
        self.mel_dir = mel_dir
        self.instrument_classes = instrument_classes
        self.instruments_mapping = instruments_mapping
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.train = train
    def __len__(self):
        return len(self.track_list)
    def __getitem__(self, idx):
        item = self.track_list[idx]
        mel_path = os.path.join(self.mel_dir, f"{item['track']}.npy")
        mel = np.load(mel_path).astype(np.float32)
        if mel.shape[1] < self.max_frames:
            pad_width = self.max_frames - mel.shape[1]
            mel = np.pad(mel, ((0,0),(0,pad_width)), mode='constant')
        else:
            mel = mel[:, :self.max_frames]
        midi = utils_ai.read_midi(item["midi"])
        frame_duration = 512 / 22050
        label = utils_ai.midi_to_label(midi, self.max_frames, frame_duration, self.instruments_mapping, self.instrument_classes)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label).float()
        return mel_tensor, label_tensor

class CRNN(nn.Module):
    def __init__(self, n_mels, n_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), padding=1), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, (3,3), padding=1), nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        self.rnn = nn.GRU((n_mels//4)*64, 128, batch_first=True, bidirectional=True, num_layers=1)
        self.fc = nn.Linear(128*2, n_classes)
    def forward(self, x):
        # x: (B, 1, n_mels, frames)
        x = self.cnn(x)  # (B, C, n_mels//4, frames//4)
        b, c, h, w = x.shape
        x = x.permute(0,3,1,2).reshape(b, w, -1)  # (B, frames//4, C*h)
        x, _ = self.rnn(x)
        out = self.fc(x)  # (B, frames//4, n_classes)
        return out