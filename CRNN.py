import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import os


class CRNNDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError("This method should be overridden by subclasses.")

# CRNN 的參數都還沒設定！
class CRNN(nn.Module):
    def __init__(self, ):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = , kernel_size =, padding = ),
            nn.BatchNorm2d(),
            nn.ReLu(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.rnn = nn.GRU(
            input_size = ,
            hidden_size = ,
            num_layers = ,
            batch_first = True,
            bidirectional = True,           
        )

        self.fc = nn.Linear(in_features = ,out_features = )
    
    def forward(self, x):
        x = self.cnn(x)
